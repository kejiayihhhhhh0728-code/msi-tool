"""RF 组合分类器与 Biomarker 筛选（批次目录适配版）。"""
from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


COORD_COLUMNS = {'relative_x', 'relative_y', 'raw_x', 'raw_y', 'he_x', 'he_y'}


@dataclass
class SamplePixels:
    sample_id: str
    sample_name: str
    X_cancer: np.ndarray
    X_para: np.ndarray
    mets: list[str]


def _fig_to_b64(fig, dpi: int = 150) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def _safe_name(name: str) -> str:
    return ''.join(c if c.isalnum() or c in ('-', '_', '.') else '_' for c in str(name))[:80] or 'tissue'


def _load_batch_meta(batch_dir: str) -> dict:
    with open(os.path.join(batch_dir, 'batch_meta.json'), 'r', encoding='utf-8') as f:
        import json
        return json.load(f)


def load_diff_table(batch_dir: str, tissue: str) -> pd.DataFrame:
    path = os.path.join(batch_dir, 'diff', _safe_name(tissue), 'differential_metabolites.csv')
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def select_diff_metabolites(diff_df: pd.DataFrame, max_features: int = 30) -> list[str]:
    if diff_df.empty or 'Metabolite' not in diff_df.columns:
        return []
    change_col = 'final_change' if 'final_change' in diff_df.columns else 'change'
    sig = diff_df[diff_df.get(change_col, 'Non') != 'Non'].copy()
    if sig.empty:
        sig = diff_df.copy()
    if 'mean_log2FC' in sig.columns:
        sig = sig.reindex(sig['mean_log2FC'].abs().sort_values(ascending=False).index)
    return sig['Metabolite'].astype(str).head(max_features).tolist()


def _mz_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in COORD_COLUMNS]


def _read_intensity(path: str, target_mets: list[str]) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path)
    cols = _mz_cols(df)
    by_exact = {str(c): c for c in cols}
    selected = []
    for met in target_mets:
        if met in by_exact:
            selected.append(by_exact[met])
            continue
        met_mz = str(met).split('|')[0]
        hit = next((c for c in cols if str(c).split('|')[0] == met_mz), None)
        if hit is not None:
            selected.append(hit)
    selected = list(dict.fromkeys(selected))
    return df[selected].astype(float), [str(c) for c in selected]


def collect_pixel_data(batch_dir: str, tissue: str, target_mets: list[str]) -> list[SamplePixels]:
    """从 samples/*/roi 下收集某 tissue 的 cancer/paracancer 像素级矩阵。"""
    meta = _load_batch_meta(batch_dir)
    samples = meta.get('samples', [])
    out: list[SamplePixels] = []
    for s in samples:
        sid = s['id']
        roi_dir = os.path.join(batch_dir, 'samples', sid, 'roi')
        pb_path = os.path.join(roi_dir, 'pseudobulk_all.csv')
        if not os.path.exists(pb_path):
            continue
        try:
            pb = pd.read_csv(pb_path)
        except Exception:
            continue
        sub = pb[pb.get('tissue_name', '') == tissue]
        if sub.empty:
            continue

        parts = {'cancer': [], 'paracancer': []}
        met_names: Optional[list[str]] = None
        for _, row in sub.iterrows():
            rtype = row.get('region_type')
            if rtype not in parts:
                continue
            stem = f"{_safe_name(row.get('tissue_name', ''))}__{_safe_name(row.get('region_name', ''))}"
            path = os.path.join(roi_dir, f'{stem}_intensity.csv')
            if not os.path.exists(path):
                # 兼容旧文件名：找包含 region_name 的 intensity csv
                rname = str(row.get('region_name', ''))
                matches = [f for f in os.listdir(roi_dir)
                           if f.endswith('_intensity.csv') and rname and rname in f]
                if matches:
                    path = os.path.join(roi_dir, matches[0])
            if not os.path.exists(path):
                continue
            try:
                Xdf, names = _read_intensity(path, target_mets)
            except Exception:
                continue
            if Xdf.empty:
                continue
            if met_names is None:
                met_names = names
            common = [m for m in met_names if m in names]
            if not common:
                continue
            parts[rtype].append(Xdf[common].values)
            met_names = common

        if parts['cancer'] and parts['paracancer'] and met_names:
            Xc = np.vstack([p[:, :len(met_names)] for p in parts['cancer']])
            Xp = np.vstack([p[:, :len(met_names)] for p in parts['paracancer']])
            out.append(SamplePixels(
                sample_id=sid,
                sample_name=s.get('name') or sid,
                X_cancer=Xc,
                X_para=Xp,
                mets=met_names,
            ))
    return out


def _common_mets(samples: list[SamplePixels]) -> list[str]:
    if not samples:
        return []
    common = set(samples[0].mets)
    for s in samples[1:]:
        common &= set(s.mets)
    return sorted(common, key=lambda x: float(str(x).split('|')[0]) if _is_float(str(x).split('|')[0]) else str(x))


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def _align(samples: list[SamplePixels], mets: list[str]) -> list[SamplePixels]:
    aligned = []
    for s in samples:
        idx = [s.mets.index(m) for m in mets]
        aligned.append(SamplePixels(s.sample_id, s.sample_name, s.X_cancer[:, idx], s.X_para[:, idx], mets))
    return aligned


def _balanced_subsample(X: np.ndarray, y: np.ndarray, max_per_class: int, rng) -> tuple[np.ndarray, np.ndarray]:
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    n = min(max_per_class, len(pos), len(neg))
    if n <= 0:
        return X, y
    idx = np.concatenate([rng.choice(pos, n, replace=False), rng.choice(neg, n, replace=False)])
    rng.shuffle(idx)
    return X[idx], y[idx]


def per_sample_aucs(samples: list[SamplePixels], mets: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """计算每个代谢物在每个样本内的像素级 AUC，返回（均值，标准差）。

    与实验记录 14 sample-AUC.py 一致：
    - 在单个样本内部用 cancer/para 像素计算 AUC
    - 若 AUC < 0.5 取 1−AUC（衡量绝对区分能力）
    - 跨样本取均值与标准差，反映区分能力的跨患者稳定性
    """
    n_mets = len(mets)
    aucs = np.full((len(samples), n_mets), np.nan, dtype=np.float64)
    for s_idx, s in enumerate(samples):
        for k in range(n_mets):
            scores = np.concatenate([s.X_cancer[:, k], s.X_para[:, k]])
            labels = np.concatenate([np.ones(len(s.X_cancer)), np.zeros(len(s.X_para))])
            mask = np.isfinite(scores)
            if mask.sum() < 4 or np.std(scores[mask]) == 0:
                continue
            try:
                auc = float(roc_auc_score(labels[mask], scores[mask]))
                if auc < 0.5:
                    auc = 1.0 - auc
                aucs[s_idx, k] = auc
            except Exception:
                continue
    mean_aucs = np.nanmean(aucs, axis=0)
    std_aucs = np.nanstd(aucs, axis=0)
    return mean_aucs, std_aucs


def rank_single_auc(samples: list[SamplePixels], mets: list[str], exclude_idx: Optional[int] = None) -> list[tuple[str, int, float]]:
    rows = []
    for k, met in enumerate(mets):
        vals, labels = [], []
        for i, s in enumerate(samples):
            if exclude_idx is not None and i == exclude_idx:
                continue
            vals += [s.X_cancer[:, k], s.X_para[:, k]]
            labels += [np.ones(len(s.X_cancer)), np.zeros(len(s.X_para))]
        score = np.concatenate(vals)
        y = np.concatenate(labels)
        mask = np.isfinite(score)
        if mask.sum() < 4 or len(np.unique(y[mask])) < 2 or np.nanstd(score[mask]) == 0:
            auc = 0.5
        else:
            auc = roc_auc_score(y[mask], score[mask])
            if auc < 0.5:
                auc = 1 - auc
        rows.append((met, k, float(auc)))
    rows.sort(key=lambda x: x[2], reverse=True)
    return rows


def nested_loso(samples: list[SamplePixels], mets: list[str], top_n: int,
                n_estimators: int = 200, max_pixels_per_class: int = 20000,
                seed: int = 42) -> tuple[np.ndarray, np.ndarray, list[tuple[list[int], np.ndarray]]]:
    labels_all, probs_all, fold_imps = [], [], []
    rng = np.random.default_rng(seed)
    if len(samples) < 2:
        return np.array([]), np.array([]), []

    for i, test in enumerate(samples):
        ranked = rank_single_auc(samples, mets, exclude_idx=i)
        actual_n = min(top_n, len(ranked))
        if actual_n < 2:
            continue
        idx = [ranked[j][1] for j in range(actual_n)]

        X_train_parts, y_train_parts = [], []
        for j, s in enumerate(samples):
            if j == i:
                continue
            X_train_parts += [s.X_cancer[:, idx], s.X_para[:, idx]]
            y_train_parts += [np.ones(len(s.X_cancer)), np.zeros(len(s.X_para))]
        X_train = np.vstack(X_train_parts)
        y_train = np.concatenate(y_train_parts)
        X_test = np.vstack([test.X_cancer[:, idx], test.X_para[:, idx]])
        y_test = np.concatenate([np.ones(len(test.X_cancer)), np.zeros(len(test.X_para))])

        tr_mask = np.all(np.isfinite(X_train), axis=1)
        te_mask = np.all(np.isfinite(X_test), axis=1)
        X_train, y_train = X_train[tr_mask], y_train[tr_mask]
        X_test, y_test = X_test[te_mask], y_test[te_mask]
        if len(X_train) < 4 or len(X_test) < 4:
            continue
        if max_pixels_per_class:
            X_train, y_train = _balanced_subsample(X_train, y_train, max_pixels_per_class, rng)
        clf = RandomForestClassifier(
            n_estimators=int(n_estimators),
            max_depth=10,
            min_samples_leaf=20,
            class_weight='balanced',
            n_jobs=-1,
            random_state=seed,
        )
        clf.fit(X_train, y_train)
        probs_all.append(clf.predict_proba(X_test)[:, 1])
        labels_all.append(y_test)
        fold_imps.append((idx, clf.feature_importances_))
    if not labels_all:
        return np.array([]), np.array([]), fold_imps
    return np.concatenate(labels_all), np.concatenate(probs_all), fold_imps


def _permute_samples(samples: list[SamplePixels], rng) -> list[SamplePixels]:
    out = []
    for s in samples:
        if rng.random() < 0.5:
            out.append(SamplePixels(s.sample_id, s.sample_name, s.X_para, s.X_cancer, s.mets))
        else:
            out.append(s)
    return out


def permutation_test(samples: list[SamplePixels], mets: list[str], top_n: int, real_auc: float,
                     n_perm: int = 200, seed: int = 42,
                     early_stop: bool = True, early_stop_min: int = 200,
                     early_stop_pval: float = 0.01) -> tuple[np.ndarray, float]:
    """样本层面 sign-flip 置换检验。

    参考实验记录 14（Stedman 2009 自适应 + Ojala & Garriga 2010 +1 校正）：
    - 默认 200 次（与该记录中 PLS-DA 置换次数一致），p 值分辨率 ≈ 0.005
    - 受样本数限制：n_samples=9 时独立配置上限 2^9=512；n_samples≥11 时上限 2000
    - 当累计 ≥ early_stop_min 且中间 p < early_stop_pval 时提前退出（节省时间）
    """
    n_samples = len(samples)
    hard_cap = min(2 ** n_samples, 2000)
    n_perm = min(int(n_perm), hard_cap)

    null = []
    for p in range(int(n_perm)):
        rng = np.random.default_rng(seed + p)
        labels, probs, _ = nested_loso(
            _permute_samples(samples, rng), mets, top_n,
            n_estimators=100, max_pixels_per_class=5000, seed=seed + p,
        )
        if len(labels) < 4 or len(np.unique(labels)) < 2:
            null.append(0.5)
        else:
            null.append(float(roc_auc_score(labels, probs)))
        # 早停：累积足够多且 p 已极小
        if early_stop and (p + 1) >= early_stop_min and (p + 1) % 50 == 0:
            arr = np.asarray(null)
            mid_p = (np.sum(arr >= real_auc) + 1) / (len(arr) + 1)
            if mid_p < early_stop_pval:
                break
    null = np.asarray(null)
    pval = float((np.sum(null >= real_auc) + 1) / (len(null) + 1)) if len(null) else 1.0
    return null, pval


def _aggregate_importance(fold_imps: list[tuple[list[int], np.ndarray]], n_mets: int) -> tuple[np.ndarray, np.ndarray]:
    s = np.zeros(n_mets)
    c = np.zeros(n_mets)
    for idx, imp in fold_imps:
        for local_i, global_i in enumerate(idx):
            s[global_i] += imp[local_i]
            c[global_i] += 1
    return np.divide(s, c, out=np.zeros_like(s), where=c > 0), c


def _plot_roc(single_best, combo_results: dict[int, tuple[float, np.ndarray, np.ndarray]], perm_pvals: dict[int, float], title: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], '--', color='#999', linewidth=1)
    if single_best:
        met, auc, fpr, tpr = single_best
        ax.plot(fpr, tpr, color='#666', linestyle='-.', label=f'Best single {met} ({auc:.3f})')
    colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a']
    for i, top_n in enumerate(sorted(combo_results)):
        auc, fpr, tpr = combo_results[top_n]
        p = perm_pvals.get(top_n)
        suffix = f', p={p:.3f}' if p is not None else ''
        ax.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=2, label=f'RF Top-{top_n} ({auc:.3f}{suffix})')
    ax.set_xlabel('1 - Specificity (FPR)')
    ax.set_ylabel('Sensitivity (TPR)')
    ax.set_title(title)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8, loc='lower right')
    ax.set_aspect('equal')
    fig.tight_layout()
    return _fig_to_b64(fig)


def _plot_auc_bar(single_rank, combo_results: dict[int, tuple[float, np.ndarray, np.ndarray]]) -> str:
    labels = [m for m, _, _ in single_rank[:10]] + [f'RF Top-{n}' for n in sorted(combo_results)]
    aucs = [a for _, _, a in single_rank[:10]] + [combo_results[n][0] for n in sorted(combo_results)]
    colors = ['#636363'] * min(10, len(single_rank)) + ['#d95f02'] * len(combo_results)
    fig, ax = plt.subplots(figsize=(9, max(4, len(labels) * 0.32)))
    y = np.arange(len(labels))
    ax.barh(y, aucs, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels([str(x)[:28] for x in labels], fontsize=8)
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel('AUC')
    ax.invert_yaxis()
    fig.tight_layout()
    return _fig_to_b64(fig)


def _plot_perm(null: np.ndarray, real_auc: float, pval: float, top_n: int) -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(null, bins=min(25, max(5, len(null))), color='#bdc3c7', edgecolor='white')
    ax.axvline(real_auc, color='#d95f02', linewidth=2, label=f'Observed {real_auc:.3f}')
    ax.set_title(f'Permutation Test RF Top-{top_n} (p={pval:.4f})')
    ax.set_xlabel('AUC')
    ax.set_ylabel('Count')
    ax.legend()
    fig.tight_layout()
    return _fig_to_b64(fig)


def build_decision_table(mets: list[str], single_rank: list[tuple[str, int, float]],
                         imp_mean: np.ndarray, imp_count: np.ndarray,
                         diff_df: pd.DataFrame, n_folds: int,
                         sample_auc_mean: Optional[np.ndarray] = None,
                         sample_auc_std: Optional[np.ndarray] = None) -> pd.DataFrame:
    auc_map = {m: auc for m, _, auc in single_rank}
    stats = diff_df.set_index('Metabolite') if 'Metabolite' in diff_df.columns else pd.DataFrame()
    rows = []
    for i, met in enumerate(mets):
        row = {'Metabolite_ID': met, 'Pixel_AUC_single': auc_map.get(met, 0.5)}
        if not stats.empty and met in stats.index:
            r = stats.loc[met]
            if isinstance(r, pd.DataFrame):
                r = r.iloc[0]
            row.update({
                'FDR': r.get('FDR', np.nan),
                'PValue': r.get('PValue', np.nan),
                'log2FC': r.get('mean_log2FC', np.nan),
                'VIP': r.get('VIP', np.nan),
                'Change': r.get('final_change', r.get('change', 'Non')),
                'Direction_Consistency': r.get('direction_consistency', np.nan),
            })
        row['RF_Importance_mean'] = imp_mean[i] if i < len(imp_mean) else 0
        row['RF_Fold_Selected'] = int(imp_count[i]) if i < len(imp_count) else 0
        row['RF_Fold_Total'] = n_folds
        if sample_auc_mean is not None and i < len(sample_auc_mean):
            row['Mean_Sample_AUC'] = float(sample_auc_mean[i]) if np.isfinite(sample_auc_mean[i]) else np.nan
        if sample_auc_std is not None and i < len(sample_auc_std):
            row['Std_Sample_AUC'] = float(sample_auc_std[i]) if np.isfinite(sample_auc_std[i]) else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    for col in ['Pixel_AUC_single', 'FDR', 'PValue', 'log2FC', 'VIP', 'Direction_Consistency',
                'RF_Importance_mean', 'Mean_Sample_AUC', 'Std_Sample_AUC']:
        if col not in df.columns:
            df[col] = np.nan
    def norm(s):
        s = pd.to_numeric(s, errors='coerce').fillna(0)
        return (s - s.min()) / (s.max() - s.min() + 1e-10)
    df['Composite_Score'] = (
        0.35 * norm(df['Pixel_AUC_single']) +
        0.20 * norm(df['log2FC'].abs()) +
        0.20 * norm(df['VIP']) +
        0.15 * norm(df['RF_Importance_mean']) +
        0.10 * norm(df['Direction_Consistency'])
    ).round(4)
    return df.sort_values('Composite_Score', ascending=False).reset_index(drop=True)


def run_classifier_pipeline(
    batch_dir: str,
    tissue: str,
    *,
    top_n_list: list[int] | None = None,
    n_estimators: int = 200,
    n_permutations: int = 200,
    max_features: int = 30,
    max_pixels_per_class: int = 20000,
) -> dict:
    diff_df = load_diff_table(batch_dir, tissue)
    target_mets = select_diff_metabolites(diff_df, max_features=max_features)
    if not target_mets:
        return {'error': '未找到模块8差异代谢物结果，请先运行模块8'}
    samples = collect_pixel_data(batch_dir, tissue, target_mets)
    if not samples:
        return {'error': '未找到可用的 ROI 像素级 cancer/paracancer CSV，请先运行模块7'}
    if len(samples) < 2:
        return {'error': '当前只有 1 个样本：无法执行 nested LOSO RF 和样本级置换检验；分类器模块至少需要 2 个配对样本。'}
    common = _common_mets(samples)
    if len(common) < 2:
        return {'error': '共同差异代谢物少于 2 个，无法训练组合分类器'}
    samples = _align(samples, common)
    ranked = rank_single_auc(samples, common)

    best_met, best_idx, best_auc = ranked[0]
    scores = np.concatenate([s.X_cancer[:, best_idx] for s in samples] + [s.X_para[:, best_idx] for s in samples])
    labels = np.concatenate([np.ones(len(s.X_cancer)) for s in samples] + [np.zeros(len(s.X_para)) for s in samples])
    mask = np.isfinite(scores)
    fpr_s, tpr_s, _ = roc_curve(labels[mask], scores[mask])
    single_best = (best_met, best_auc, fpr_s, tpr_s)

    combo_results = {}
    perm_pvals = {}
    perm_imgs = {}
    best_fold_imps = []
    best_auc_combo = -1
    best_top = None
    if len(samples) >= 2:
        for top_n in top_n_list or [3, 5, 10, 15]:
            actual = min(int(top_n), len(common))
            if actual < 2:
                continue
            y, prob, imps = nested_loso(samples, common, actual, n_estimators, max_pixels_per_class)
            if len(y) < 4 or len(np.unique(y)) < 2:
                continue
            auc = float(roc_auc_score(y, prob))
            fpr, tpr, _ = roc_curve(y, prob)
            combo_results[actual] = (auc, fpr, tpr)
            null, pval = permutation_test(samples, common, actual, auc, n_permutations)
            perm_pvals[actual] = pval
            perm_imgs[actual] = _plot_perm(null, auc, pval, actual)
            if auc > best_auc_combo:
                best_auc_combo = auc
                best_fold_imps = imps
                best_top = actual

    imp_mean, imp_count = _aggregate_importance(best_fold_imps, len(common)) if best_fold_imps else (np.zeros(len(common)), np.zeros(len(common)))
    # 计算逐样本 AUC（均值与标准差），用于评估代谢物区分能力的跨样本稳定性
    samp_auc_mean, samp_auc_std = per_sample_aucs(samples, common)
    decision = build_decision_table(
        common, ranked, imp_mean, imp_count, diff_df, len(samples),
        sample_auc_mean=samp_auc_mean, sample_auc_std=samp_auc_std,
    )

    return {
        'samples': samples,
        'mets': common,
        'ranked_single': ranked,
        'combo_results': combo_results,
        'perm_pvals': perm_pvals,
        'decision_df': decision,
        'roc_img': _plot_roc(single_best, combo_results, perm_pvals, f'Random Forest ROC - {tissue}'),
        'auc_img': _plot_auc_bar(ranked, combo_results),
        'perm_imgs': perm_imgs,
        'best_top_n': best_top,
        'n_samples': len(samples),
        'n_metabolites': len(common),
        'n_cancer_px': int(sum(len(s.X_cancer) for s in samples)),
        'n_para_px': int(sum(len(s.X_para) for s in samples)),
    }
