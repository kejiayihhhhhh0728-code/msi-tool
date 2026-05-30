"""
差异代谢物分析（不依赖 Flask）
==============================

完整移植自 5.差异代谢物分析/差异分析_完整pipeline.py，主要变化：
- 去掉硬编码路径，所有 I/O 改成函数参数
- plt.savefig 输出统一改为 base64 PNG（让 Flask 路由直接当 JSON 返回）
- 算法逻辑：配对 Wilcoxon (n>=6) / 配对 t-test (n<6) + BH FDR + PLS-DA VIP
- log2FC 使用数据自适应 pseudocount；PLS-DA 置换检验基于 paired LOSO CV-AUC

核心入口: run_full_pipeline(cancer_mat, para_mat, ...)
"""
from __future__ import annotations

import base64
import io
import json
import os
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, ttest_rel, ttest_ind
from scipy import stats as sp_stats
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = [
    'Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'Arial Unicode MS', 'DejaVu Sans'
]
matplotlib.rcParams['axes.unicode_minus'] = False
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
import seaborn as sns


# ─── 默认参数（与参考脚本一致）────────────────────────────────────────
DEFAULT_FC_CUTOFF = float(np.log2(1.5))     # ~0.585
DEFAULT_P_CUTOFF = 0.05
DEFAULT_VIP_CUTOFF = 1.0
MIN_SAMPLES_WILCOXON = 6                    # < 6 自动退化为 paired t-test
DEFAULT_PLSDA_PERMUTATION_N = 200
DEFAULT_PLSDA_COMPONENTS = 2
DEFAULT_LOG2FC_EPSILON_QUANTILE = 0.01
PIXEL_COORD_COLUMNS = {'relative_x', 'relative_y', 'raw_x', 'raw_y', 'he_x', 'he_y'}


# ─── 通用工具 ─────────────────────────────────────────────────────────

def _fig_to_b64(fig, dpi: int = 150) -> str:
    """matplotlib Figure -> base64 PNG（无 data: 前缀）"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor() or 'white')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _ggplot_style(ax, title: str = ''):
    """模拟 R ggplot2 theme_bw"""
    ax.set_facecolor('white')
    ax.grid(True, color='gray', alpha=0.3, linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#333333')
        spine.set_linewidth(0.8)
    ax.axhline(0, color='black', linewidth=0.5, zorder=1)
    ax.axvline(0, color='black', linewidth=0.5, zorder=1)
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=12)


def _draw_confidence_ellipse(x, y, ax, **kw):
    """95% 置信椭圆（t 分布校正，对应 R stat_ellipse(type='t')）"""
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    mean_x, mean_y = float(np.mean(x)), float(np.mean(y))
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    angle = float(np.degrees(np.arctan2(*eigvecs[:, 0][::-1])))
    n = len(x)
    t_val = sp_stats.t.ppf(0.975, df=n - 1)
    scale = t_val * np.sqrt(1 + 1.0 / n)
    width = 2 * scale * np.sqrt(eigvals[0])
    height = 2 * scale * np.sqrt(eigvals[1])
    ellipse = Ellipse(
        xy=(mean_x, mean_y), width=width, height=height, angle=angle,
        facecolor=kw.get('facecolor', 'none'),
        edgecolor=kw.get('edgecolor', 'black'),
        linestyle=kw.get('linestyle', '--'),
        linewidth=kw.get('linewidth', 1.2),
        alpha=kw.get('alpha', 0.6),
    )
    ax.add_patch(ellipse)


def estimate_log2fc_epsilon(
    cancer_mat: pd.DataFrame,
    para_mat: pd.DataFrame,
    q: float = DEFAULT_LOG2FC_EPSILON_QUANTILE,
) -> float:
    """
    用空间代谢组 pseudo-bulk 正强度的低分位数估计 log2FC pseudocount。
    这相当于用数据自适应的检测限替代固定极小值，避免 0 附近的 fold-change 被放大。
    """
    vals = np.concatenate([
        cancer_mat.to_numpy(dtype=float, copy=False).ravel(),
        para_mat.to_numpy(dtype=float, copy=False).ravel(),
    ])
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if len(vals) == 0:
        return float(np.finfo(float).eps)
    q = float(np.clip(q, 0.0, 1.0))
    eps = float(np.quantile(vals, q))
    return float(max(eps, np.finfo(float).eps))


def _safe_name(name: str) -> str:
    return ''.join(c if c.isalnum() or c in ('-', '_', '.') else '_' for c in str(name))[:80] or 'tissue'


# ─── 中位数中心化 ──────────────────────────────────────────────────────

def median_center_correction(cancer_mat: pd.DataFrame, para_mat: pd.DataFrame
                             ) -> tuple[pd.DataFrame, dict]:
    """
    对每个样本独立估计 paracancer→cancer 的整体中位数偏移，把 paracancer 抬升
    （或压低）至与 cancer 同一基线。消除 TIC 系统性偏差。
    """
    para_corrected = para_mat.copy()
    corrections = {}
    eps = estimate_log2fc_epsilon(cancer_mat, para_mat)
    for col in para_mat.columns:
        c_vals = cancer_mat[col].values.astype(float)
        p_vals = para_mat[col].values.astype(float)
        raw_fc = np.log2((c_vals + eps) / (p_vals + eps))
        med = np.nanmedian(raw_fc[np.isfinite(raw_fc)])
        if np.isfinite(med):
            para_corrected[col] = p_vals * (2 ** med)
            corrections[col] = float(med)
    return para_corrected, corrections


# ─── 差异分析（配对 Wilcoxon / t-test + BH FDR）─────────────────────────

def differential_analysis(
    cancer_mat: pd.DataFrame,
    para_mat: pd.DataFrame,
    fc_cutoff: float = DEFAULT_FC_CUTOFF,
    p_cutoff: float = DEFAULT_P_CUTOFF,
    min_samples_wilcoxon: int = MIN_SAMPLES_WILCOXON,
) -> tuple[pd.DataFrame, str]:
    """
    返回 (diff_df, test_used)。
    test_used 为字符串说明：'wilcoxon (paired)' 或 'ttest_rel (paired)'。
    """
    mets = cancer_mat.index.tolist()
    sample_names = cancer_mat.columns.tolist()
    n_samples = len(sample_names)
    use_wilcoxon = n_samples >= min_samples_wilcoxon
    test_used = 'wilcoxon (paired)' if use_wilcoxon else 'ttest_rel (paired)'
    if n_samples == 1:
        test_used = 'ttest_rel (paired; n=1 descriptive)'
    eps = estimate_log2fc_epsilon(cancer_mat, para_mat)

    results = []
    for met in mets:
        c_vals = cancer_mat.loc[met].values.astype(float)
        p_vals = para_mat.loc[met].values.astype(float)
        per_sample_fc = np.log2((c_vals + eps) / (p_vals + eps))
        mean_log2fc = float(np.nanmean(per_sample_fc))

        diffs = c_vals - p_vals
        if np.all(diffs == 0):
            pval = 1.0
        elif n_samples == 1:
            # 单对样本没有 t 检验自由度；保留 log2FC 描述量，p 值标记为不可估计。
            pval = np.nan
        elif use_wilcoxon:
            try:
                _, pval = wilcoxon(c_vals, p_vals, alternative='two-sided')
            except ValueError:
                pval = 1.0
        else:
            try:
                _, pval = ttest_rel(c_vals, p_vals)
            except Exception:
                pval = 1.0
            if not np.isfinite(pval):
                pval = 1.0

        valid_fc = per_sample_fc[np.isfinite(per_sample_fc)]
        if n_samples == 1:
            consistency = np.nan
        elif len(valid_fc) > 0:
            direction = np.sign(np.nanmean(valid_fc))
            consistency = float(np.mean(np.sign(valid_fc) == direction))
        else:
            consistency = 0.0

        row = {
            'Metabolite': met,
            'mean_log2FC': mean_log2fc,
            'cancer_grand_mean': float(np.mean(c_vals)),
            'paracancer_grand_mean': float(np.mean(p_vals)),
            'PValue': float(pval) if np.isfinite(pval) else np.nan,
            'n_samples': int(n_samples),
            'direction_consistency': consistency,
        }
        for j, name in enumerate(sample_names):
            row[f'{name}_log2FC'] = float(per_sample_fc[j]) if np.isfinite(per_sample_fc[j]) else np.nan
        results.append(row)

    df = pd.DataFrame(results)
    if df.empty:
        return df, test_used

    _, fdr, _, _ = multipletests(df['PValue'].fillna(1).values, method='fdr_bh')
    df['FDR'] = fdr
    if n_samples == 1:
        df['FDR'] = np.nan

    df['change'] = 'Non'
    df.loc[(df['FDR'] < p_cutoff) & (df['mean_log2FC'] >=  fc_cutoff), 'change'] = 'Up'
    df.loc[(df['FDR'] < p_cutoff) & (df['mean_log2FC'] <= -fc_cutoff), 'change'] = 'Down'

    return df.sort_values('FDR').reset_index(drop=True), test_used


# ─── 单样本像素级差异分析 ──────────────────────────────────────────────

def _load_batch_meta(batch_dir: str) -> dict:
    with open(os.path.join(batch_dir, 'batch_meta.json'), 'r', encoding='utf-8') as f:
        return json.load(f)


def _mz_sort_key(s):
    try:
        return (0, float(str(s).split('|')[0]))
    except (ValueError, TypeError):
        return (1, str(s))


def _find_roi_intensity_path(roi_dir: str, tissue_name: str, region_name: str) -> str | None:
    stem = f"{_safe_name(tissue_name)}__{_safe_name(region_name)}"
    path = os.path.join(roi_dir, f'{stem}_intensity.csv')
    if os.path.exists(path):
        return path
    if not os.path.isdir(roi_dir):
        return None
    matches = [
        f for f in os.listdir(roi_dir)
        if f.endswith('_intensity.csv') and str(region_name) and str(region_name) in f
    ]
    return os.path.join(roi_dir, matches[0]) if matches else None


def _collect_single_sample_pixel_mats(batch_dir: str, tissue_name: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    收集单个样本内 cancer/paracancer ROI 的像素级矩阵。
    返回 cancer_pixels × mz、paracancer_pixels × mz，以及样本信息。
    """
    bmeta = _load_batch_meta(batch_dir)
    for s in bmeta.get('samples', []):
        sid = s['id']
        roi_dir = os.path.join(batch_dir, 'samples', sid, 'roi')
        pb_path = os.path.join(roi_dir, 'pseudobulk_all.csv')
        if not os.path.exists(pb_path):
            continue
        try:
            pb = pd.read_csv(pb_path)
        except Exception:
            continue
        if not {'tissue_name', 'region_type', 'region_name'}.issubset(pb.columns):
            continue
        sub = pb[pb['tissue_name'] == tissue_name]
        if sub.empty or not {'cancer', 'paracancer'}.issubset(set(sub['region_type'])):
            continue

        parts = {'cancer': [], 'paracancer': []}
        for _, row in sub.iterrows():
            rtype = row.get('region_type')
            if rtype not in parts:
                continue
            path = _find_roi_intensity_path(roi_dir, tissue_name, row.get('region_name', ''))
            if not path:
                continue
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            mz_cols = [c for c in df.columns if c not in PIXEL_COORD_COLUMNS]
            if mz_cols:
                parts[rtype].append(df[mz_cols].astype(float))

        if parts['cancer'] and parts['paracancer']:
            common = set(parts['cancer'][0].columns)
            for df in parts['cancer'][1:] + parts['paracancer']:
                common &= set(df.columns)
            common = sorted(common, key=_mz_sort_key)
            if not common:
                continue
            cancer_px = pd.concat([df[common] for df in parts['cancer']], ignore_index=True)
            para_px = pd.concat([df[common] for df in parts['paracancer']], ignore_index=True)
            return cancer_px, para_px, {
                'sample_id': sid,
                'sample_name': s.get('name') or sid,
                'n_cancer_pixels': int(len(cancer_px)),
                'n_paracancer_pixels': int(len(para_px)),
            }

    return pd.DataFrame(), pd.DataFrame(), {}


def single_sample_pixel_analysis(
    cancer_px: pd.DataFrame,
    para_px: pd.DataFrame,
    fc_cutoff: float = DEFAULT_FC_CUTOFF,
    p_cutoff: float = DEFAULT_P_CUTOFF,
) -> pd.DataFrame:
    """单样本 ROI 像素级 Welch t-test + BH-FDR。"""
    common = [c for c in cancer_px.columns if c in set(para_px.columns)]
    eps_vals = np.concatenate([
        cancer_px[common].to_numpy(dtype=float, copy=False).ravel(),
        para_px[common].to_numpy(dtype=float, copy=False).ravel(),
    ])
    eps_vals = eps_vals[np.isfinite(eps_vals) & (eps_vals > 0)]
    eps = float(np.quantile(eps_vals, DEFAULT_LOG2FC_EPSILON_QUANTILE)) if len(eps_vals) else float(np.finfo(float).eps)
    eps = max(eps, float(np.finfo(float).eps))

    rows = []
    for met in common:
        c = cancer_px[met].to_numpy(dtype=float)
        p = para_px[met].to_numpy(dtype=float)
        c = c[np.isfinite(c)]
        p = p[np.isfinite(p)]
        if len(c) == 0 or len(p) == 0:
            continue
        mean_log2fc = float(np.log2((np.mean(c) + eps) / (np.mean(p) + eps)))
        if len(c) >= 2 and len(p) >= 2 and (np.nanstd(c) > 0 or np.nanstd(p) > 0):
            _, pval = ttest_ind(c, p, equal_var=False, nan_policy='omit')
            if not np.isfinite(pval):
                pval = 1.0
        else:
            pval = 1.0
        try:
            labels = np.concatenate([np.ones(len(c)), np.zeros(len(p))])
            scores = np.concatenate([c, p])
            auc = float(roc_auc_score(labels, scores)) if np.nanstd(scores) > 0 else 0.5
        except Exception:
            auc = 0.5
        abs_auc = max(auc, 1.0 - auc)
        rows.append({
            'Metabolite': met,
            'mean_log2FC': mean_log2fc,
            'cancer_grand_mean': float(np.mean(c)),
            'paracancer_grand_mean': float(np.mean(p)),
            'PValue': float(pval),
            'n_samples': 1,
            'n_cancer_pixels': int(len(c)),
            'n_paracancer_pixels': int(len(p)),
            'Pixel_AUC': auc,
            'Pixel_AUC_abs': abs_auc,
            'Effect_rank_biserial': float(2 * auc - 1),
            'direction_consistency': abs_auc,
            'VIP': np.nan,
            'pixel_log2FC': mean_log2fc,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    _, fdr, _, _ = multipletests(df['PValue'].fillna(1).values, method='fdr_bh')
    df['FDR'] = fdr
    df['change'] = 'Non'
    df.loc[(df['FDR'] < p_cutoff) & (df['mean_log2FC'] >= fc_cutoff), 'change'] = 'Up'
    df.loc[(df['FDR'] < p_cutoff) & (df['mean_log2FC'] <= -fc_cutoff), 'change'] = 'Down'
    df['final_change'] = df['change']
    return df.sort_values(['FDR', 'mean_log2FC'], ascending=[True, False]).reset_index(drop=True)


def run_single_sample_pixel_pipeline(
    batch_dir: str,
    tissue_name: str,
    *,
    fc_cutoff: float = DEFAULT_FC_CUTOFF,
    p_cutoff: float = DEFAULT_P_CUTOFF,
    vip_cutoff: float = DEFAULT_VIP_CUTOFF,
) -> dict:
    cancer_px, para_px, sample_info = _collect_single_sample_pixel_mats(batch_dir, tissue_name)
    if cancer_px.empty or para_px.empty:
        return {'error': '未找到该 tissue 的单样本 cancer/paracancer ROI 像素级数据'}
    diff_df = single_sample_pixel_analysis(cancer_px, para_px, fc_cutoff=fc_cutoff, p_cutoff=p_cutoff)
    if diff_df.empty:
        return {'error': '单样本像素级差异分析无结果'}

    plot_df = diff_df.copy()
    volcano_img = plot_volcano(
        plot_df,
        title=f'{tissue_name} pixel-level',
        vip_series=None,
        fc_cutoff=fc_cutoff,
        p_cutoff=p_cutoff,
        vip_cutoff=vip_cutoff,
    )
    n_up = int((diff_df['final_change'] == 'Up').sum())
    n_down = int((diff_df['final_change'] == 'Down').sum())
    return {
        'diff_df': diff_df,
        'test_used': 'Welch t-test (single-sample pixel-level)',
        'median_corrections': {},
        'pca_img': None,
        'plsda_img': None,
        'plsda_perm_img': None,
        'plsda_r2': None,
        'plsda_cv_auc': None,
        'plsda_q2': None,
        'plsda_perm_pval': None,
        'volcano_img': volcano_img,
        'fc_heatmap_img': None,
        'n_up': n_up,
        'n_down': n_down,
        'n_total': int(len(diff_df)),
        'warning': (
            '当前只有 1 个样本：已切换为单样本 ROI 像素级 Welch t-test + BH-FDR。'
            '该结果表示本切片内 cancer/paracancer 像素分布差异，不是多样本配对 pseudo-bulk 结论；VIP/PLS-DA 已跳过。'
        ),
        'sample_info': sample_info,
    }


# ─── PCA 可视化 ───────────────────────────────────────────────────────

def run_pca(cancer_mat: pd.DataFrame, para_mat: pd.DataFrame,
            group_name: str = '') -> tuple[str, pd.DataFrame]:
    sample_names = cancer_mat.columns.tolist()

    data_list, labels, groups = [], [], []
    for name in sample_names:
        data_list.append(cancer_mat[name].values)
        labels.append(f'{name}_C'); groups.append('Cancer')
        data_list.append(para_mat[name].values)
        labels.append(f'{name}_P'); groups.append('Paracancer')

    X = np.array(data_list)
    X_log = np.log2(X + 1)
    X_scaled = StandardScaler().fit_transform(X_log)

    n_comp = min(2, X_scaled.shape[0] - 1, X_scaled.shape[1])
    if n_comp < 1:
        return '', pd.DataFrame()

    pca = PCA(n_components=n_comp)
    scores = pca.fit_transform(X_scaled)
    var_ratio = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    color_map = {'Cancer': '#D95F02', 'Paracancer': '#1B9E77'}
    for grp in ('Cancer', 'Paracancer'):
        mask = np.array([g == grp for g in groups])
        sx = scores[mask, 0]
        sy = scores[mask, 1] if n_comp >= 2 else np.zeros(int(mask.sum()))
        ax.scatter(sx, sy, c=color_map[grp], s=80, edgecolors='white',
                   linewidth=0.5, label=grp, alpha=0.85, zorder=3)
        if mask.sum() >= 3 and n_comp >= 2:
            _draw_confidence_ellipse(sx, sy, ax, edgecolor=color_map[grp])

    # 同样本 cancer/paracancer 用虚线连
    for i, name in enumerate(sample_names):
        c_x = float(scores[2 * i, 0])
        c_y = float(scores[2 * i, 1]) if n_comp >= 2 else 0.0
        p_x = float(scores[2 * i + 1, 0])
        p_y = float(scores[2 * i + 1, 1]) if n_comp >= 2 else 0.0
        ax.plot([c_x, p_x], [c_y, p_y], color='gray', linestyle=':',
                linewidth=0.6, alpha=0.5, zorder=1)

    ax.set_xlabel(f'Dim 1 ({var_ratio[0]:.2f}%)', fontsize=12)
    if n_comp >= 2:
        ax.set_ylabel(f'Dim 2 ({var_ratio[1]:.2f}%)', fontsize=12)
    _ggplot_style(ax, title=f'PCA — {group_name}' if group_name else 'PCA')
    ax.legend(fontsize=10, frameon=True, edgecolor='gray')

    img_b64 = _fig_to_b64(fig)
    scores_df = pd.DataFrame(scores, columns=[f'PC{i+1}' for i in range(n_comp)])
    scores_df['label'] = labels
    scores_df['group'] = groups
    return img_b64, scores_df


# ─── PLS-DA + VIP ──────────────────────────────────────────────────────

def compute_vip(pls_model, X: np.ndarray) -> np.ndarray:
    W = pls_model.x_weights_
    T = pls_model.x_scores_
    Q = pls_model.y_loadings_
    p, h = W.shape
    SS = np.array([(T[:, a] @ T[:, a]) * (Q[:, a] @ Q[:, a]) for a in range(h)])
    SS_total = float(np.sum(SS))
    vip = np.zeros(p)
    for j in range(p):
        weighted = float(np.sum(SS * (W[j, :] ** 2)))
        vip[j] = float(np.sqrt(p * weighted / SS_total)) if SS_total > 0 else 0.0
    return vip


def _plsda_paired_loso_metrics(
    X_log: np.ndarray,
    y: np.ndarray,
    pair_ids: np.ndarray,
    n_components: int,
) -> tuple[Optional[float], Optional[float]]:
    """按病人/样本成对留一验证 PLS-DA，返回 (CV-AUC, Q2)。"""
    unique_pairs = np.unique(pair_ids)
    if len(unique_pairs) < 2:
        return None, None

    y_true, y_pred = [], []
    for pair_id in unique_pairs:
        train_mask = pair_ids != pair_id
        test_mask = pair_ids == pair_id
        if len(np.unique(y[train_mask])) < 2:
            continue

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_log[train_mask])
        X_test = scaler.transform(X_log[test_mask])
        max_comp = min(int(n_components), X_train.shape[0] - 1, X_train.shape[1])
        if max_comp < 1:
            continue

        pls = PLSRegression(n_components=max_comp, scale=False)
        pls.fit(X_train, y[train_mask])
        pred = pls.predict(X_test).ravel()
        y_true.extend(y[test_mask].tolist())
        y_pred.extend(pred.tolist())

    if len(y_true) < 4 or len(np.unique(y_true)) < 2:
        return None, None

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    try:
        cv_auc = float(roc_auc_score(y_true, y_pred))
    except Exception:
        cv_auc = None

    press = float(np.sum((y_true - y_pred) ** 2))
    tss = float(np.sum((y_true - np.mean(y_true)) ** 2))
    q2 = float(1.0 - press / tss) if tss > 0 else None
    return cv_auc, q2


def _paired_label_permutation(y: np.ndarray, pair_ids: np.ndarray, rng) -> np.ndarray:
    """成对设计的标签置换：每对样本内随机交换 cancer/paracancer 标签。"""
    y_perm = y.copy()
    for pair_id in np.unique(pair_ids):
        idx = np.where(pair_ids == pair_id)[0]
        if len(idx) == 2 and rng.random() < 0.5:
            y_perm[idx] = y_perm[idx[::-1]]
    return y_perm


def run_plsda(
    cancer_mat: pd.DataFrame,
    para_mat: pd.DataFrame,
    group_name: str = '',
    n_components: int = DEFAULT_PLSDA_COMPONENTS,
    n_permutation: int = DEFAULT_PLSDA_PERMUTATION_N,
    vip_cutoff: float = DEFAULT_VIP_CUTOFF,
    rng_seed: Optional[int] = None,
) -> dict:
    """
    返回 dict:
      {
        'vip_series'  : pd.Series (index=metabolites),
        'r2'          : float,
        'cv_auc'      : float,
        'q2'          : float,
        'perm_pval'   : float,
        'plsda_img'   : str (base64 PNG，得分图+VIP top30),
        'perm_img'    : str (base64 PNG，置换检验直方图),
      }
    样本数太少时返回 vip_series 为空，其他字段为 None。
    """
    sample_names = cancer_mat.columns.tolist()
    mets = cancer_mat.index.tolist()
    if len(sample_names) < 2:
        return {'vip_series': pd.Series(dtype=float),
                'r2': None, 'cv_auc': None, 'q2': None, 'perm_pval': None,
                'plsda_img': None, 'perm_img': None}

    data_list, y_labels, point_labels = [], [], []
    pair_ids = []
    for pair_idx, name in enumerate(sample_names):
        data_list.append(cancer_mat[name].values); y_labels.append(1); point_labels.append(f'{name}_C')
        pair_ids.append(pair_idx)
        data_list.append(para_mat[name].values);   y_labels.append(0); point_labels.append(f'{name}_P')
        pair_ids.append(pair_idx)
    X = np.array(data_list)
    y = np.array(y_labels)
    pair_ids = np.array(pair_ids)

    X_log = np.log2(X + 1)
    X_scaled = StandardScaler().fit_transform(X_log)

    max_comp = min(n_components, X_scaled.shape[0] - 1, X_scaled.shape[1])
    if max_comp < 1:
        return {'vip_series': pd.Series(dtype=float),
                'r2': None, 'cv_auc': None, 'q2': None, 'perm_pval': None,
                'plsda_img': None, 'perm_img': None}

    pls = PLSRegression(n_components=max_comp, scale=False)
    pls.fit(X_scaled, y)

    vip = compute_vip(pls, X_scaled)
    vip_series = pd.Series(vip, index=mets, name='VIP')

    # R²X% per component
    T = pls.x_scores_
    ss_total = float(np.sum(X_scaled ** 2))
    r2x_per_comp = []
    for a in range(max_comp):
        t_a = T[:, a:a + 1]
        p_a = (t_a.T @ X_scaled) / (t_a.T @ t_a)
        X_hat_a = t_a @ p_a
        r2x_per_comp.append(float(np.sum(X_hat_a ** 2) / ss_total * 100))

    # 得分图 + VIP Top30
    fig, axes = plt.subplots(
        1, 2, figsize=(21, 8), gridspec_kw={'width_ratios': [1.05, 1.45]}
    )
    ax = axes[0]
    colors_map = {1: '#D95F02', 0: '#1B9E77'}
    labels_map = {1: 'Cancer', 0: 'Paracancer'}
    for grp in (1, 0):
        idx = np.where(y == grp)[0]
        sc = T[idx]
        y_vals = sc[:, 1] if max_comp >= 2 else np.zeros(len(idx))
        ax.scatter(sc[:, 0], y_vals, c=colors_map[grp], s=80,
                   edgecolors='white', linewidth=0.5,
                   label=labels_map[grp], alpha=0.85, zorder=3)
        if len(idx) >= 3 and max_comp >= 2:
            _draw_confidence_ellipse(sc[:, 0], y_vals, ax, edgecolor=colors_map[grp])
        for i in idx:
            yi = T[i, 1] if max_comp >= 2 else 0
            ax.annotate(point_labels[i], (T[i, 0], yi),
                        fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                  edgecolor='gray', alpha=0.7),
                        ha='center', va='bottom',
                        xytext=(0, 6), textcoords='offset points')
    ax.set_xlabel(f'X-variate 1 ({r2x_per_comp[0]:.1f}%)', fontsize=13)
    if max_comp >= 2:
        ax.set_ylabel(f'X-variate 2 ({r2x_per_comp[1]:.1f}%)', fontsize=13)
    _ggplot_style(ax, title=f'PLS-DA — {group_name}' if group_name else 'PLS-DA')
    ax.legend(fontsize=10, frameon=True, edgecolor='gray')

    # VIP Top 30
    ax2 = axes[1]
    top_n = min(30, len(vip_series))
    vip_top = vip_series.nlargest(top_n).sort_values(ascending=True)
    bar_colors = ['#D95F02' if v > vip_cutoff else '#95A5A6' for v in vip_top.values]
    ax2.barh(range(top_n), vip_top.values, color=bar_colors,
             edgecolor='white', height=0.7)
    ax2.set_yticks(range(top_n))
    ax2.set_yticklabels(vip_top.index, fontsize=8.5)
    ax2.axvline(x=vip_cutoff, color='red', linestyle='--', linewidth=1,
                alpha=0.7, label=f'VIP = {vip_cutoff}')
    ax2.set_xlabel('VIP Score', fontsize=13)
    _ggplot_style(ax2, title=f'Top {top_n} VIP')
    ax2.legend(fontsize=9, loc='lower right')
    fig.tight_layout()
    plsda_img = _fig_to_b64(fig, dpi=180)

    # 置换检验：以 paired LOSO CV-AUC 为主要统计量，Q2 作为交叉验证预测能力辅助指标。
    rng = np.random.default_rng(rng_seed)
    real_r2 = float(pls.score(X_scaled, y))
    real_cv_auc, real_q2 = _plsda_paired_loso_metrics(X_log, y, pair_ids, max_comp)
    perm_cv_aucs = []
    if int(n_permutation) > 0:
        for _ in range(int(n_permutation)):
            y_perm = _paired_label_permutation(y, pair_ids, rng)
            try:
                perm_auc, _ = _plsda_paired_loso_metrics(X_log, y_perm, pair_ids, max_comp)
                if perm_auc is not None:
                    perm_cv_aucs.append(float(perm_auc))
            except Exception:
                continue
    perm_cv_aucs = np.array(perm_cv_aucs) if perm_cv_aucs else np.array([0.5])
    if real_cv_auc is None or int(n_permutation) <= 0:
        perm_pval = None
    else:
        perm_pval = float((np.sum(perm_cv_aucs >= real_cv_auc) + 1) / (len(perm_cv_aucs) + 1))

    fig2, ax3 = plt.subplots(figsize=(6, 4.5))
    ax3.hist(perm_cv_aucs, bins=30, color='#BDC3C7', edgecolor='white',
             alpha=0.8, label='Permuted')
    if real_cv_auc is not None:
        ax3.axvline(real_cv_auc, color='#D95F02', linewidth=2,
                    label=f'Observed CV-AUC = {real_cv_auc:.3f}')
    ax3.set_xlabel('Paired LOSO CV-AUC', fontsize=12)
    ax3.set_ylabel('Count', fontsize=12)
    p_label = f'{perm_pval:.4f}' if perm_pval is not None else 'NA'
    q2_label = f'{real_q2:.3f}' if real_q2 is not None else 'NA'
    _ggplot_style(ax3, title=f'PLS-DA Permutation\nCV-AUC p = {p_label}, Q2 = {q2_label}')
    ax3.legend(fontsize=10)
    fig2.tight_layout()
    perm_img = _fig_to_b64(fig2)

    return {
        'vip_series' : vip_series,
        'r2'         : real_r2,
        'cv_auc'     : real_cv_auc,
        'q2'         : real_q2,
        'perm_pval'  : perm_pval,
        'plsda_img'  : plsda_img,
        'perm_img'   : perm_img,
    }


# ─── 火山图 ───────────────────────────────────────────────────────────

def plot_volcano(
    diff_df: pd.DataFrame,
    title: str,
    vip_series: Optional[pd.Series] = None,
    fc_cutoff: float = DEFAULT_FC_CUTOFF,
    p_cutoff: float = DEFAULT_P_CUTOFF,
    vip_cutoff: float = DEFAULT_VIP_CUTOFF,
) -> str:
    fig, ax = plt.subplots(figsize=(9, 7))
    neg_log_fdr = -np.log10(diff_df['FDR'].clip(lower=1e-300)).fillna(0)

    colors, sizes = [], []
    for _, row in diff_df.iterrows():
        if row['change'] == 'Up':
            colors.append('#FB0F15')
        elif row['change'] == 'Down':
            colors.append('#0709F7')
        else:
            colors.append('#999999')
        if vip_series is not None:
            v = vip_series.get(row['Metabolite'], 0)
            sizes.append(50 if v > vip_cutoff else 15)
        else:
            sizes.append(20)

    ax.scatter(diff_df['mean_log2FC'], neg_log_fdr,
               c=colors, s=sizes, alpha=0.5, edgecolors='none', zorder=2)
    ax.axhline(-np.log10(p_cutoff), color='black', linestyle='--', linewidth=0.6, alpha=0.6)
    ax.axvline( fc_cutoff, color='black', linestyle='--', linewidth=0.6, alpha=0.6)
    ax.axvline(-fc_cutoff, color='black', linestyle='--', linewidth=0.6, alpha=0.6)

    if vip_series is not None:
        sig = diff_df[diff_df['change'] != 'Non'].copy()
        sig['VIP'] = sig['Metabolite'].map(vip_series)
        sig = sig[sig['VIP'] > vip_cutoff].nlargest(15, 'VIP')
        for _, row in sig.iterrows():
            ax.annotate(row['Metabolite'],
                        (row['mean_log2FC'], -np.log10(max(row['FDR'], 1e-300))),
                        fontsize=5.5, alpha=0.8, ha='center', va='bottom')

    n_up = int((diff_df['change'] == 'Up').sum())
    n_down = int((diff_df['change'] == 'Down').sum())
    ax.set_xlabel(r'$\log_2$(FC)', fontsize=13)
    ax.set_ylabel(r'$-\log_{10}$(FDR)', fontsize=13)
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FB0F15',
               markersize=8, label=f'Up ({n_up})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#0709F7',
               markersize=8, label=f'Down ({n_down})'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#999999',
               markersize=8, label='Stable'),
    ]
    if vip_series is not None:
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                   markersize=10, label=f'VIP > {vip_cutoff} (larger dot)'))
    ax.legend(handles=legend_elements, fontsize=9, loc='best')
    _ggplot_style(ax, title=f'{title}\nUp: {n_up}  Down: {n_down}')
    fig.tight_layout()
    return _fig_to_b64(fig)


# ─── FC 热图 ─────────────────────────────────────────────────────────

def plot_fc_heatmap(diff_df: pd.DataFrame, sample_names: list,
                    title: str, max_mets: int = 50) -> Optional[str]:
    sig = diff_df[diff_df['change'] != 'Non'].copy()
    if sig.empty:
        return None
    fc_cols = [c for c in diff_df.columns if c.endswith('_log2FC')]
    if not fc_cols:
        return None
    sig = sig.reindex(sig['mean_log2FC'].abs().sort_values(ascending=False).index)
    sig = sig.head(max_mets)
    plot_df = sig.set_index('Metabolite')[fc_cols].copy()
    plot_df.columns = [c.replace('_log2FC', '') for c in plot_df.columns]

    height = max(4.0, len(plot_df) * 0.22)
    fig, ax = plt.subplots(figsize=(max(6, len(sample_names) * 0.5), height))
    sns.heatmap(plot_df, cmap='RdBu_r', center=0, ax=ax,
                cbar_kws={'label': 'log2 FC (per sample)'},
                linewidths=0.3, linecolor='white')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.tick_params(axis='y', labelsize=6)
    fig.tight_layout()
    return _fig_to_b64(fig)


# ─── 综合筛选 ────────────────────────────────────────────────────────

def comprehensive_filter(
    diff_df: pd.DataFrame,
    vip_series: Optional[pd.Series],
    fc_cutoff: float = DEFAULT_FC_CUTOFF,
    p_cutoff: float = DEFAULT_P_CUTOFF,
    vip_cutoff: float = DEFAULT_VIP_CUTOFF,
) -> pd.DataFrame:
    """三重筛选：FDR + |log2FC| + VIP"""
    df = diff_df.copy()
    if vip_series is not None and not vip_series.empty:
        df['VIP'] = df['Metabolite'].map(vip_series).fillna(0)
        require_vip = True
    else:
        df['VIP'] = 0
        require_vip = False

    df['final_change'] = 'Non'
    cond_up   = (df['FDR'] < p_cutoff) & (df['mean_log2FC'] >=  fc_cutoff)
    cond_down = (df['FDR'] < p_cutoff) & (df['mean_log2FC'] <= -fc_cutoff)
    if require_vip:
        cond_up   &= (df['VIP'] > vip_cutoff)
        cond_down &= (df['VIP'] > vip_cutoff)
    df.loc[cond_up,   'final_change'] = 'Up'
    df.loc[cond_down, 'final_change'] = 'Down'
    return df


# ─── 总入口：跑完整流程 ──────────────────────────────────────────────

def run_full_pipeline(
    cancer_mat: pd.DataFrame,
    para_mat: pd.DataFrame,
    *,
    group_name: str = '',
    fc_cutoff: float = DEFAULT_FC_CUTOFF,
    p_cutoff: float = DEFAULT_P_CUTOFF,
    vip_cutoff: float = DEFAULT_VIP_CUTOFF,
    do_median_correction: bool = True,
    do_pca: bool = True,
    do_plsda: bool = True,
    plsda_n_perm: int = DEFAULT_PLSDA_PERMUTATION_N,
    rng_seed: Optional[int] = None,
) -> dict:
    """
    跑完整差异分析流程。返回结果 dict 中所有图都是 base64 PNG。
    """
    if cancer_mat.shape[1] < 1:
        return {'error': '需要至少 1 个配对样本'}
    warnings = []
    if cancer_mat.shape[1] == 1:
        warnings.append('当前只有 1 个配对样本：差异分析仅报告描述性 log2FC，PValue/FDR 不用于显著性判断，PLS-DA/VIP 已跳过。')

    corrections = {}
    if do_median_correction:
        para_mat, corrections = median_center_correction(cancer_mat, para_mat)

    diff_df, test_used = differential_analysis(
        cancer_mat, para_mat,
        fc_cutoff=fc_cutoff, p_cutoff=p_cutoff,
    )
    if diff_df.empty:
        return {'error': '差异分析无结果（样本太少？）'}

    vip_series = pd.Series(dtype=float)
    plsda_img = perm_img = None
    r2 = cv_auc = q2 = perm_pval = None
    if do_plsda:
        try:
            plsda_out = run_plsda(
                cancer_mat, para_mat,
                group_name=group_name,
                vip_cutoff=vip_cutoff,
                n_permutation=plsda_n_perm,
                rng_seed=rng_seed,
            )
            vip_series = plsda_out['vip_series']
            plsda_img = plsda_out['plsda_img']
            perm_img = plsda_out['perm_img']
            r2 = plsda_out['r2']
            cv_auc = plsda_out['cv_auc']
            q2 = plsda_out['q2']
            perm_pval = plsda_out['perm_pval']
        except Exception:
            plsda_img = None

    pca_img = None
    if do_pca:
        try:
            pca_img, _ = run_pca(cancer_mat, para_mat, group_name=group_name)
        except Exception:
            pca_img = None

    diff_df = comprehensive_filter(
        diff_df, vip_series,
        fc_cutoff=fc_cutoff, p_cutoff=p_cutoff, vip_cutoff=vip_cutoff,
    )
    if cancer_mat.shape[1] == 1:
        diff_df[['PValue', 'FDR', 'VIP', 'direction_consistency']] = np.nan
        diff_df['final_change'] = 'Non'

    plot_df = diff_df.copy()
    plot_df['change'] = plot_df['final_change']
    volcano_img = plot_volcano(
        plot_df, title=group_name or 'Volcano',
        vip_series=vip_series if not vip_series.empty else None,
        fc_cutoff=fc_cutoff, p_cutoff=p_cutoff, vip_cutoff=vip_cutoff,
    )

    fc_heatmap_img = plot_fc_heatmap(
        plot_df, sample_names=list(cancer_mat.columns),
        title=f'差异代谢物 FC 热图 — {group_name}' if group_name else '差异代谢物 FC 热图',
    )

    n_up = int((diff_df['final_change'] == 'Up').sum())
    n_down = int((diff_df['final_change'] == 'Down').sum())

    return {
        'diff_df'           : diff_df,
        'test_used'         : test_used,
        'median_corrections': corrections,
        'pca_img'           : pca_img,
        'plsda_img'         : plsda_img,
        'plsda_perm_img'    : perm_img,
        'plsda_r2'          : r2,
        'plsda_cv_auc'      : cv_auc,
        'plsda_q2'          : q2,
        'plsda_perm_pval'   : perm_pval,
        'volcano_img'       : volcano_img,
        'fc_heatmap_img'    : fc_heatmap_img,
        'n_up'              : n_up,
        'n_down'            : n_down,
        'n_total'           : int(len(diff_df)),
        'warning'           : ' '.join(warnings) if warnings else None,
    }
