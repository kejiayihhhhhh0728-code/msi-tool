"""通路富集分析（离线 GMT / 超几何检验）。"""
from __future__ import annotations

import base64
import io
import os
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _fig_to_b64(fig, dpi: int = 150) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def safe_name(name: str) -> str:
    return ''.join(c if c.isalnum() or c in ('-', '_', '.') else '_' for c in str(name))[:80] or 'tissue'


def read_gmt(gmt_path: str) -> dict[str, list[str]]:
    pathways = {}
    with open(gmt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split('\t') if p.strip()]
            if len(parts) >= 2:
                pathways[parts[0]] = list(dict.fromkeys(parts[1:]))
    return pathways


def _load_batch_meta(batch_dir: str) -> dict:
    import json
    with open(os.path.join(batch_dir, 'batch_meta.json'), 'r', encoding='utf-8') as f:
        return json.load(f)


def load_diff_table(batch_dir: str, tissue: str) -> pd.DataFrame:
    path = os.path.join(batch_dir, 'diff', safe_name(tissue), 'differential_metabolites.csv')
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def collect_annotation_map(batch_dir: str) -> dict[str, set[str]]:
    """
    聚合当前批次所有样本 annotation_results.csv，返回 mz字符串 -> KEGG集合。
    兼容几种可能来源：
    - annotation_results.csv 中存在 KEGG 列
    - db_id 本身就是 KEGG Compound ID (Cxxxxx)
    - db_id / db_name 中含 Cxxxxx
    """
    meta = _load_batch_meta(batch_dir)
    mapping: dict[str, set[str]] = defaultdict(set)
    for s in meta.get('samples', []):
        path = os.path.join(batch_dir, 'samples', s['id'], 'annotation_results.csv')
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if 'mz_observed' not in df.columns:
            continue
        for _, row in df.iterrows():
            try:
                mz_key = f"{float(row['mz_observed']):.4f}"
            except Exception:
                continue
            candidates = []
            for col in ['KEGG', 'kegg', 'db_id', 'db_name']:
                if col in df.columns and pd.notna(row.get(col)):
                    candidates.extend(str(row.get(col)).replace(';', ',').split(','))
            for c in candidates:
                c = c.strip()
                if c.startswith('C') and len(c) >= 5:
                    mapping[mz_key].add(c[:6] if len(c) > 6 and c[1:6].isdigit() else c)
    return mapping


def add_kegg_to_diff(diff_df: pd.DataFrame, mz_to_kegg: dict[str, set[str]]) -> pd.DataFrame:
    df = diff_df.copy()
    kegg_values = []
    for met in df.get('Metabolite', pd.Series(dtype=str)).astype(str):
        mz_key = met.split('|')[0]
        try:
            mz_key = f"{float(mz_key):.4f}"
        except Exception:
            pass
        ids = sorted(mz_to_kegg.get(mz_key, []))
        kegg_values.append(';'.join(ids))
    df['KEGG'] = kegg_values
    return df


def _split_kegg(series: pd.Series) -> list[str]:
    out = []
    for val in series.dropna().astype(str):
        for item in val.replace(',', ';').split(';'):
            item = item.strip()
            if item.startswith('C') and len(item) >= 5:
                out.append(item)
    return list(dict.fromkeys(out))


def filter_diff(df: pd.DataFrame, fc_cutoff: float, fdr_cutoff: float, use_vip: bool,
                vip_cutoff: float) -> pd.DataFrame:
    out = df.copy()
    if 'mean_log2FC' in out.columns and 'FDR' in out.columns:
        out['pathway_change'] = 'Non'
        up = (out['FDR'] < fdr_cutoff) & (out['mean_log2FC'] >= fc_cutoff)
        down = (out['FDR'] < fdr_cutoff) & (out['mean_log2FC'] <= -fc_cutoff)
        if use_vip and 'VIP' in out.columns:
            up &= out['VIP'] >= vip_cutoff
            down &= out['VIP'] >= vip_cutoff
        out.loc[up, 'pathway_change'] = 'Up'
        out.loc[down, 'pathway_change'] = 'Down'
    else:
        col = 'final_change' if 'final_change' in out.columns else 'change'
        out['pathway_change'] = out.get(col, 'Non')
    return out


def enrich_kegg(gene_list: list[str], background: list[str], gmt: dict[str, list[str]]) -> pd.DataFrame:
    genes = set(gene_list)
    bg = set(background)
    if not genes or not bg:
        return pd.DataFrame()
    rows = []
    M = len(bg)
    N = len(genes & bg)
    for term, members in gmt.items():
        members_bg = set(members) & bg
        hits = genes & members_bg
        if not hits:
            continue
        K = len(members_bg)
        x = len(hits)
        pval = float(hypergeom.sf(x - 1, M, K, N))
        rows.append({
            'Term': term,
            'P-value': pval,
            'Overlap': f'{x}/{K}',
            'Hits': ';'.join(sorted(hits)),
            'Input_size': N,
            'Pathway_size': K,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    _, fdr, _, _ = multipletests(df['P-value'].values, method='fdr_bh')
    df['Adjusted P-value'] = fdr
    df['-log10(FDR)'] = -np.log10(df['Adjusted P-value'].clip(lower=1e-300))
    return df.sort_values('Adjusted P-value').reset_index(drop=True)


def plot_dotplot(result_df: pd.DataFrame, title: str, p_cutoff: float = 0.05,
                 top_n: int = 20) -> Optional[str]:
    if result_df.empty:
        return None
    df = result_df.head(top_n).copy()
    df = df.sort_values('Adjusted P-value', ascending=False)
    sizes = df['Overlap'].apply(lambda x: int(str(x).split('/')[0]) if '/' in str(x) else 1)
    colors = -np.log10(df['Adjusted P-value'].clip(lower=1e-300))
    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.42)))
    sc = ax.scatter(colors, range(len(df)), s=sizes * 55 + 25,
                    c=colors, cmap='RdYlBu_r', edgecolors='black', linewidths=0.4)
    ax.axvline(-np.log10(p_cutoff), color='gray', linestyle='--', linewidth=1)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['Term'], fontsize=8)
    ax.set_xlabel('-log10(Adjusted P-value)')
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, label='-log10(FDR)', shrink=0.7)
    fig.tight_layout()
    return _fig_to_b64(fig)


def run_pathway_pipeline(
    batch_dir: str,
    tissue: str,
    gmt_path: str,
    *,
    fc_cutoff: float = float(np.log2(1.2)),
    fdr_cutoff: float = 0.05,
    use_vip: bool = False,
    vip_cutoff: float = 1.0,
    pathway_sig_cutoff: float = 0.05,
) -> dict:
    diff_df = load_diff_table(batch_dir, tissue)
    if diff_df.empty:
        return {'error': '未找到模块8差异代谢物结果，请先运行模块8'}
    gmt = read_gmt(gmt_path)
    mz_to_kegg = collect_annotation_map(batch_dir)
    diff_df = add_kegg_to_diff(diff_df, mz_to_kegg)
    diff_df = filter_diff(diff_df, fc_cutoff, fdr_cutoff, use_vip, vip_cutoff)

    background = _split_kegg(diff_df['KEGG']) if 'KEGG' in diff_df.columns else []
    if not background:
        return {'error': '没有可用 KEGG 映射。请确认模块2注释结果包含 KEGG，或数据库 id 使用 KEGG Compound ID。'}

    outputs = {}
    sig_terms = set()
    for change in ['Up', 'Down']:
        sub = diff_df[diff_df['pathway_change'] == change]
        genes = _split_kegg(sub['KEGG']) if 'KEGG' in sub.columns else []
        enr = enrich_kegg(genes, background, gmt)
        img = plot_dotplot(enr, f'{tissue} {change} pathway enrichment', p_cutoff=pathway_sig_cutoff)
        if not enr.empty:
            sig_terms.update(enr[enr['Adjusted P-value'] < pathway_sig_cutoff]['Term'].tolist())
        outputs[change] = {
            'genes': genes,
            'result_df': enr,
            'dotplot_img': img,
        }

    return {
        'diff_df': diff_df,
        'background_size': len(background),
        'mapped_metabolites': int((diff_df['KEGG'].astype(str).str.len() > 0).sum()),
        'gmt_pathways': len(gmt),
        'outputs': outputs,
        'sig_pathways': sorted(sig_terms),
    }


# ============================================================
# Mode B：空间通路活性热图（参考实验记录 16）
# ============================================================

def _sample_kegg_mz_map(batch_dir: str, sample_id: str) -> list[tuple[float, set[str]]]:
    """读取单样本 annotation_results.csv，构建 [(mz_observed, {KEGG…}), …]。"""
    path = os.path.join(batch_dir, 'samples', sample_id, 'annotation_results.csv')
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    if 'mz_observed' not in df.columns:
        return []
    rows = []
    for _, row in df.iterrows():
        try:
            mz = float(row['mz_observed'])
        except Exception:
            continue
        kegg = set()
        for col in ('KEGG', 'kegg', 'db_id', 'db_name'):
            if col in df.columns and pd.notna(row.get(col)):
                for c in str(row.get(col)).replace(';', ',').split(','):
                    c = c.strip()
                    if c.startswith('C') and len(c) >= 5:
                        kegg.add(c[:6] if len(c) > 6 and c[1:6].isdigit() else c)
        if kegg:
            rows.append((mz, kegg))
    return rows


def _match_mz_to_kegg(adata_mz: np.ndarray,
                      kegg_mz_pairs: list[tuple[float, set[str]]],
                      tolerance: float = 0.005) -> dict[str, list[int]]:
    """
    将 h5ad 中每个 m/z 通道按容差匹配到 KEGG ID。
    返回 {KEGG: [col_idx, …]}。
    """
    if not kegg_mz_pairs or len(adata_mz) == 0:
        return {}
    kegg_to_cols: dict[str, list[int]] = defaultdict(list)
    sorted_mz = np.asarray([p[0] for p in kegg_mz_pairs])
    sorted_kegg = [p[1] for p in kegg_mz_pairs]
    order = np.argsort(sorted_mz)
    sorted_mz = sorted_mz[order]
    sorted_kegg = [sorted_kegg[i] for i in order]
    for col_idx, mz_val in enumerate(adata_mz):
        if not np.isfinite(mz_val):
            continue
        i = int(np.searchsorted(sorted_mz, mz_val))
        for j in (i - 1, i):
            if 0 <= j < len(sorted_mz) and abs(sorted_mz[j] - mz_val) <= tolerance:
                for k in sorted_kegg[j]:
                    kegg_to_cols[k].append(col_idx)
    # 去重
    return {k: sorted(set(v)) for k, v in kegg_to_cols.items()}


def _plot_spatial_pathway(scores: np.ndarray,
                          coords: np.ndarray,
                          title: str,
                          n_metabolites: int) -> str:
    """绘制单条通路的空间活性热图，返回 base64。"""
    xs = coords[:, 0].astype(float)
    ys = coords[:, 1].astype(float)
    # 用相对坐标网格（保留小数精度）：先量化到整数像素栅格
    xi = np.rint(xs - xs.min()).astype(int)
    yi = np.rint(ys - ys.min()).astype(int)
    grid = np.full((yi.max() + 1, xi.max() + 1), np.nan, dtype=np.float32)
    for x, y, s in zip(xi, yi, scores):
        if np.isfinite(s):
            grid[y, x] = s
    # 颜色范围：取 z-score 的对称分位数
    finite = scores[np.isfinite(scores)]
    if finite.size:
        vmax = float(np.nanpercentile(np.abs(finite), 95))
        vmax = max(vmax, 1e-6)
    else:
        vmax = 1.0
    fig, ax = plt.subplots(figsize=(7, 6), facecolor='white')
    im = ax.imshow(grid, cmap='RdYlBu_r', aspect='equal',
                   interpolation='nearest', vmin=-vmax, vmax=vmax)
    ax.invert_yaxis()
    ax.set_title(f'{title}\nPathway activity (mean z-score, {n_metabolites} metabolites)',
                 fontsize=10, fontweight='bold')
    ax.axis('off')
    cbar = plt.colorbar(im, ax=ax, shrink=0.75)
    cbar.set_label('Mean z-score', fontsize=9)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _compute_pathway_scores_for_sample(adata,
                                       gmt: dict[str, list[str]],
                                       pathway_names: list[str],
                                       kegg_to_cols: dict[str, list[int]]) -> dict[str, tuple[np.ndarray, int]]:
    """对单样本计算每条通路的 per-pixel 活性分数 (mean z-score)."""
    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)  # (n_pixels, n_mz)

    # TIC 归一化（防御性，h5ad 通常已归一化但再做一次无副作用）
    tic = X.sum(axis=1)
    tic_safe = np.where(tic > 0, tic, 1.0)
    median_tic = float(np.nanmedian(tic_safe[tic > 0])) if (tic > 0).any() else 1.0
    X_norm = X / tic_safe[:, None] * median_tic

    # 跨像素的代谢物 z-score
    mu = X_norm.mean(axis=0)
    sigma = X_norm.std(axis=0)
    sigma = np.where(sigma > 0, sigma, 1.0)
    Z = (X_norm - mu) / sigma  # (n_pixels, n_mz)

    out: dict[str, tuple[np.ndarray, int]] = {}
    for pw in pathway_names:
        members = gmt.get(pw, [])
        if not members:
            continue
        cols: list[int] = []
        for k in members:
            cols.extend(kegg_to_cols.get(k, []))
        cols = sorted(set(cols))
        if not cols:
            continue
        scores = Z[:, cols].mean(axis=1)
        out[pw] = (scores.astype(np.float32), len(cols))
    return out


def run_spatial_pathway_pipeline(
    batch_dir: str,
    pathway_names: list[str],
    gmt_path: str,
    *,
    sample_ids: Optional[list[str]] = None,
    mz_tolerance: float = 0.005,
    save_dir: Optional[str] = None,
) -> dict:
    """
    Mode B：将显著通路映射回每个样本的全像素空间，绘制活性热图。

    参考实验记录 16：TIC 归一化 → per-metabolite z-score → 通路成员均值 → 二维热图。

    返回结构：
    {
        'samples': [
            {
                'sample_id': str, 'sample_name': str,
                'pathways': {pathway_name: {'img': b64_png, 'n_metabolites': int}},
                'n_pathways': int, 'n_pixels': int,
            }, ...
        ],
        'pathway_overlap': pd.DataFrame,  # 通路-代谢物匹配概览
        'errors': [...],
    }
    """
    import anndata as ad

    if not pathway_names:
        return {'error': '未指定要绘制的通路（请先运行模式 A 找到显著通路）'}

    gmt = read_gmt(gmt_path)
    meta = _load_batch_meta(batch_dir)
    samples = meta.get('samples', [])
    if sample_ids:
        wanted = set(sample_ids)
        samples = [s for s in samples if s['id'] in wanted]
    if not samples:
        return {'error': '未找到可用样本，请检查批次配置'}

    sample_results = []
    overlap_rows = []
    errors: list[str] = []

    for s in samples:
        sid = s['id']
        sample_dir = os.path.join(batch_dir, 'samples', sid)
        # 找该样本的 h5ad（norm_tic > norm_rms > raw）
        h5ad_path = None
        for name in ('norm_tic.h5ad', 'norm_rms.h5ad', 'raw.h5ad'):
            p = os.path.join(sample_dir, name)
            if os.path.exists(p):
                h5ad_path = p
                break
        if not h5ad_path:
            errors.append(f'{sid}: 未找到 h5ad')
            continue
        try:
            adata = ad.read_h5ad(h5ad_path)
        except Exception as e:
            errors.append(f'{sid}: 读取 h5ad 失败 ({e})')
            continue

        # m/z 通道值
        var = adata.var
        if 'm/z' in var.columns:
            mz_arr = pd.to_numeric(var['m/z'], errors='coerce').to_numpy()
        else:
            mz_arr = pd.to_numeric(pd.Series(var.index), errors='coerce').to_numpy()
        if not np.isfinite(mz_arr).any():
            errors.append(f'{sid}: 无法解析 m/z 通道')
            continue

        # 坐标
        if 'spatial' in adata.obsm:
            coords = np.asarray(adata.obsm['spatial'], dtype=float)
        elif {'relative_x', 'relative_y'}.issubset(adata.obs.columns):
            coords = adata.obs[['relative_x', 'relative_y']].to_numpy(dtype=float)
        else:
            errors.append(f'{sid}: 无空间坐标')
            continue

        # KEGG 映射
        kegg_pairs = _sample_kegg_mz_map(batch_dir, sid)
        kegg_to_cols = _match_mz_to_kegg(mz_arr, kegg_pairs, tolerance=mz_tolerance)
        if not kegg_to_cols:
            errors.append(f'{sid}: 无 KEGG 注释命中（请先运行模块 2 注释）')
            continue

        scores_by_pw = _compute_pathway_scores_for_sample(adata, gmt, pathway_names, kegg_to_cols)
        if not scores_by_pw:
            errors.append(f'{sid}: 通路与 m/z 无交集')
            continue

        pw_outputs: dict[str, dict] = {}
        for pw, (scores, n_mets) in scores_by_pw.items():
            title = f'{s.get("name", sid)} | {pw}'
            img_b64 = _plot_spatial_pathway(scores, coords, title, n_mets)
            pw_outputs[pw] = {'img': img_b64, 'n_metabolites': int(n_mets)}
            overlap_rows.append({
                'sample_id': sid,
                'sample_name': s.get('name', sid),
                'pathway': pw,
                'n_metabolites_matched': int(n_mets),
            })
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                out_path = os.path.join(save_dir, f'{sid}_{safe_name(pw)}_spatial.png')
                try:
                    with open(out_path, 'wb') as f:
                        f.write(base64.b64decode(img_b64))
                except Exception:
                    pass

        sample_results.append({
            'sample_id': sid,
            'sample_name': s.get('name', sid),
            'pathways': pw_outputs,
            'n_pathways': len(pw_outputs),
            'n_pixels': int(adata.n_obs),
        })

    return {
        'samples': sample_results,
        'pathway_overlap': pd.DataFrame(overlap_rows),
        'errors': errors,
        'n_samples': len(sample_results),
    }

