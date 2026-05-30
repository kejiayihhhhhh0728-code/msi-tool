"""差异代谢物空间热图（批次 ROI 像素 CSV 适配版）。"""
from __future__ import annotations

import base64
import io
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


COORD_COLUMNS = {'relative_x', 'relative_y', 'raw_x', 'raw_y', 'he_x', 'he_y'}


def fig_to_base64(fig, dpi: int = 150) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def safe_name(name: str) -> str:
    return ''.join(c if c.isalnum() or c in ('-', '_', '.') else '_' for c in str(name))[:80] or 'tissue'


def load_diff_metabolites(batch_dir: str, tissue: str, max_items: int = 100) -> pd.DataFrame:
    path = os.path.join(batch_dir, 'diff', safe_name(tissue), 'differential_metabolites.csv')
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if 'Metabolite' not in df.columns:
        return pd.DataFrame()
    change_col = 'final_change' if 'final_change' in df.columns else 'change'
    sig = df[df.get(change_col, 'Non') != 'Non'].copy()
    if sig.empty:
        sig = df.copy()
    if 'mean_log2FC' in sig.columns:
        sig = sig.reindex(sig['mean_log2FC'].abs().sort_values(ascending=False).index)
    return sig.head(max_items)


def _load_batch_meta(batch_dir: str) -> dict:
    import json
    with open(os.path.join(batch_dir, 'batch_meta.json'), 'r', encoding='utf-8') as f:
        return json.load(f)


def _find_col(df: pd.DataFrame, met: str) -> str | None:
    cols = [c for c in df.columns if c not in COORD_COLUMNS]
    if met in cols:
        return met
    mz = str(met).split('|')[0]
    return next((c for c in cols if str(c).split('|')[0] == mz), None)


def _find_sample_h5ad(sample_dir: str) -> str | None:
    for name in ('norm_tic.h5ad', 'norm_rms.h5ad', 'raw.h5ad'):
        path = os.path.join(sample_dir, name)
        if os.path.exists(path):
            return path
    return None


def _as_float_mz(value) -> float | None:
    try:
        return float(str(value).split('|')[0])
    except (TypeError, ValueError):
        return None


def _match_adata_metabolite(adata, metabolite: str, tolerance: float = 0.005) -> int | None:
    names = [str(x) for x in adata.var_names]
    if metabolite in names:
        return names.index(metabolite)
    target = _as_float_mz(metabolite)
    if target is None:
        mz = str(metabolite).split('|')[0]
        return next((i for i, n in enumerate(names) if str(n).split('|')[0] == mz), None)

    best_idx = None
    best_diff = float(tolerance)
    for i, name in enumerate(names):
        mz_val = _as_float_mz(name)
        if mz_val is None and 'm/z' in adata.var.columns:
            mz_val = _as_float_mz(adata.var.iloc[i].get('m/z'))
        if mz_val is None:
            continue
        diff = abs(mz_val - target)
        if diff <= best_diff:
            best_idx = i
            best_diff = diff
    return best_idx


def _adata_coords(adata) -> tuple[np.ndarray, np.ndarray] | None:
    obs = adata.obs
    if {'raw_x', 'raw_y'}.issubset(obs.columns):
        return obs['raw_x'].to_numpy(dtype=float), obs['raw_y'].to_numpy(dtype=float)
    if {'relative_x', 'relative_y'}.issubset(obs.columns):
        return obs['relative_x'].to_numpy(dtype=float), obs['relative_y'].to_numpy(dtype=float)
    if 'spatial' in adata.obsm:
        coords = np.asarray(adata.obsm['spatial'], dtype=float)
        if coords.ndim == 2 and coords.shape[1] >= 2:
            return coords[:, 0], coords[:, 1]
    return None


def collect_metabolite_whole_slice_pixels(
    batch_dir: str,
    tissue: str,
    metabolite: str,
    *,
    mz_tolerance: float = 0.005,
) -> list[dict]:
    """从每个样本的整张切片 h5ad 读取目标代谢物空间强度。"""
    try:
        import anndata as ad
    except Exception:
        return []

    meta = _load_batch_meta(batch_dir)
    out = []
    for s in meta.get('samples', []):
        sid = s['id']
        sample_dir = os.path.join(batch_dir, 'samples', sid)
        h5ad_path = _find_sample_h5ad(sample_dir)
        if not h5ad_path:
            continue
        try:
            adata = ad.read_h5ad(h5ad_path)
        except Exception:
            continue
        col_idx = _match_adata_metabolite(adata, metabolite, tolerance=mz_tolerance)
        coords = _adata_coords(adata)
        if col_idx is None or coords is None:
            continue
        x, y = coords
        try:
            col = adata.X[:, col_idx]
            if hasattr(col, 'toarray'):
                intensity = np.asarray(col.toarray()).ravel()
            else:
                intensity = np.asarray(col).ravel()
        except Exception:
            continue
        if len(intensity) != len(x):
            continue
        out.append({
            'sample_id': sid,
            'sample_name': s.get('name') or sid,
            'region_name': 'Whole slice',
            'region_type': tissue or '',
            'x': x,
            'y': y,
            'intensity': intensity.astype(float),
        })
    return out


def collect_metabolite_pixels(batch_dir: str, tissue: str, metabolite: str) -> list[dict]:
    meta = _load_batch_meta(batch_dir)
    out = []
    for s in meta.get('samples', []):
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
        for _, row in sub.iterrows():
            stem = f"{safe_name(row.get('tissue_name', ''))}__{safe_name(row.get('region_name', ''))}"
            path = os.path.join(roi_dir, f'{stem}_intensity.csv')
            if not os.path.exists(path):
                rname = str(row.get('region_name', ''))
                matches = [f for f in os.listdir(roi_dir)
                           if f.endswith('_intensity.csv') and rname and rname in f]
                if matches:
                    path = os.path.join(roi_dir, matches[0])
            if not os.path.exists(path):
                continue
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            col = _find_col(df, metabolite)
            if col is None:
                continue
            x_col = 'he_x' if 'he_x' in df.columns else ('raw_x' if 'raw_x' in df.columns else 'relative_x')
            y_col = 'he_y' if 'he_y' in df.columns else ('raw_y' if 'raw_y' in df.columns else 'relative_y')
            if x_col not in df.columns or y_col not in df.columns:
                continue
            out.append({
                'sample_id': sid,
                'sample_name': s.get('name') or sid,
                'region_name': row.get('region_name', ''),
                'region_type': row.get('region_type', ''),
                'x': df[x_col].astype(float).values,
                'y': df[y_col].astype(float).values,
                'intensity': df[col].astype(float).values,
            })
    return out


def render_batch_heatmap(items: list[dict], metabolite: str, *, colormap: str = 'magma',
                         max_panels: int = 24) -> str | None:
    if not items:
        return None
    items = items[:max_panels]
    vals = np.concatenate([it['intensity'][np.isfinite(it['intensity'])] for it in items])
    if vals.size == 0:
        return None
    vmin, vmax = np.nanpercentile(vals, [2, 98])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals) + 1e-9)

    n = len(items)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows), squeeze=False)
    axes_flat = axes.ravel()
    last_sc = None
    for ax, it in zip(axes_flat, items):
        mask = np.isfinite(it['intensity']) & np.isfinite(it['x']) & np.isfinite(it['y'])
        last_sc = ax.scatter(it['x'][mask], it['y'][mask], c=it['intensity'][mask],
                             s=5, cmap=colormap, vmin=vmin, vmax=vmax, linewidths=0)
        ax.set_aspect('equal', adjustable='box')
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        region = f" / {it['region_name']}" if it.get('region_name') else ''
        ax.set_title(f"{it['sample_name']}{region}", fontsize=9)
    for ax in axes_flat[len(items):]:
        ax.axis('off')
    fig.suptitle(str(metabolite), fontsize=12, fontweight='bold')
    if last_sc is not None:
        fig.colorbar(last_sc, ax=axes_flat.tolist(), shrink=0.75, label='Intensity')
    return fig_to_base64(fig)
