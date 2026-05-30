"""
核心计算：MSI 数据读取与归一化
迁移自 MSIpro/datapre.py，去除 Flask 依赖，可单独测试。
"""
import csv
import io
import base64

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ─── 数据读取 ────────────────────────────────────────────────

def _detect_format(ms_file: str) -> str:
    """
    自动检测 MSI 数据文件格式。

    返回
    ----
    'xlsx_txt' : Tab 分隔，行=像素，列=m/z，首行以 'Spot index' 开头
                 （由 xlsx_to_dpm_txt.py 导出的格式）
    'mz_csv'   : 分号分隔，行=m/z，列=spot，含以 'm/z' 开头的标题行
                 （原始 MSIpro 格式）
    """
    with open(ms_file, 'r', encoding='utf-8', errors='replace') as f:
        first_line = f.readline()
    first_cell = first_line.split('\t')[0].strip().strip('"')
    if first_cell.lower().startswith('spot'):
        return 'xlsx_txt'
    return 'mz_csv'


def _read_xlsx_txt(ms_file: str, resolution: float) -> tuple:
    """
    读取 xlsx_to_dpm_txt.py 导出的 Tab 分隔文件。

    格式：首行 = "Spot index\\tx\\ty\\tmz1\\tmz2..."
          数据行：spot_idx, x, y, intensity1, intensity2, ...

    返回 (X: np.ndarray[n_pixels, n_mz], df_spot, mz_names)
    """
    df = pd.read_csv(ms_file, sep='\t', engine='python', encoding='utf-8')

    # 去掉列名两端的引号（部分导出会带引号）
    df.columns = [c.strip().strip('"') for c in df.columns]

    # 找 x / y 列（大小写不敏感）
    col_lower = {c.lower(): c for c in df.columns}
    x_col = col_lower.get('x', None)
    y_col = col_lower.get('y', None)
    if x_col is None or y_col is None:
        raise ValueError("xlsx_txt 格式中未找到 'x' / 'y' 列，请检查文件。")

    # m/z 强度列 = 除 Spot index / x / y 之外的数值列
    skip = {'spot index', 'x', 'y'}
    mz_cols = [c for c in df.columns if c.lower() not in skip]

    X = df[mz_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    coords = df[[x_col, y_col]].astype(float)
    df_spot = pd.DataFrame({
        'relative_x': ((coords[x_col] - coords[x_col].min()) / resolution).round(4),
        'relative_y': ((coords[y_col] - coords[y_col].min()) / resolution).round(4),
        'raw_x': coords[x_col].values,
        'raw_y': coords[y_col].values,
    })

    return X, df_spot, mz_cols


def ms_to_anndata(ms_file: str, spot_file: str | None,
                  chunk_size: int = 10000,
                  resolution: float = 20,
                  save_path: str = None) -> ad.AnnData:
    """
    将 MSI 数据文件整合为 AnnData 对象。支持两种输入格式：

    格式 A — xlsx_txt（xlsx_to_dpm_txt.py 导出）
        - Tab 分隔，行=像素，列=m/z，首行含 'Spot index / x / y'
        - x / y 坐标内嵌，spot_file 可传 None

    格式 B — mz_csv（原始 MSIpro 格式）
        - 分号分隔，行=m/z，列=spot，含 'm/z' 标题行
        - 需同时提供 spot_file（分号分隔，含 Spot index / x / y）

    参数
    ----
    ms_file    : MSI 数据文件路径
    spot_file  : spot 坐标文件路径（格式 B 必填，格式 A 可为 None）
    chunk_size : 格式 B 分块读取行数，默认 10000
    resolution : 空间分辨率（µm），用于计算 relative_x/y，默认 20 µm
    save_path  : 若提供则保存为 .h5ad
    """
    fmt = _detect_format(ms_file)

    if fmt == 'xlsx_txt':
        # ── 格式 A：Tab 分隔，行=像素 ──
        X_data, obs_data, mz_names = _read_xlsx_txt(ms_file, resolution)
        var_data = pd.DataFrame({'m/z': mz_names}, index=mz_names)

    else:
        # ── 格式 B：分号分隔，行=m/z ──
        if spot_file is None:
            raise ValueError("mz_csv 格式需要同时提供 spot 坐标文件。")

        data_chunks = []
        header = None
        start_data = False

        with open(ms_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';')
            chunk = []

            for row in reader:
                if row and row[0].startswith("m/z"):
                    start_data = True
                    header = row
                    continue

                if start_data:
                    if len(row) < len(header):
                        if chunk:
                            chunk[-1].extend(row)
                        else:
                            chunk.append(row)
                    else:
                        chunk.append(row)

                    if len(chunk) >= chunk_size:
                        df_chunk = pd.DataFrame(chunk, columns=header)
                        for col in df_chunk.columns[1:]:
                            df_chunk[col] = pd.to_numeric(df_chunk[col], errors='coerce')
                        data_chunks.append(df_chunk)
                        chunk = []

            if chunk:
                df_chunk = pd.DataFrame(chunk, columns=header)
                for col in df_chunk.columns[1:]:
                    df_chunk[col] = pd.to_numeric(df_chunk[col], errors='coerce')
                data_chunks.append(df_chunk)

        df_ms = pd.concat(data_chunks, ignore_index=True)
        df_ms_t = df_ms.set_index('m/z').T
        # 清理 spot 名，后续与 spot_file 做严格对齐
        df_ms_t.index = pd.Index([str(v).strip().strip('"') for v in df_ms_t.index], name='spot_id')

        with open(spot_file, 'r') as f:
            reader = csv.reader(f, delimiter=';')
            spot_header = None
            for row in reader:
                if row and row[0].startswith("Spot"):
                    spot_header = row
                    break
            data = [row for row in reader]

        df_spot_raw = pd.DataFrame(data, columns=spot_header)
        df_spot_raw.columns = [str(c).strip().strip('"') for c in df_spot_raw.columns]

        # 识别 Spot index 列并做对齐（关键：按 spot_id 对齐坐标与强度矩阵）
        spot_col = None
        for c in df_spot_raw.columns:
            cl = c.lower().replace('_', ' ').strip()
            if cl.startswith('spot'):
                spot_col = c
                break
        if spot_col is None:
            raise ValueError("spot 文件中未找到 Spot index 列，无法与 MSI 强度矩阵对齐。")

        df_spot_raw[spot_col] = df_spot_raw[spot_col].astype(str).str.strip().str.strip('"')
        df_spot_raw['x'] = pd.to_numeric(df_spot_raw['x'], errors='coerce')
        df_spot_raw['y'] = pd.to_numeric(df_spot_raw['y'], errors='coerce')

        spot_aligned = (
            df_spot_raw
            .set_index(spot_col)
            .reindex(df_ms_t.index)
        )

        missing = spot_aligned[['x', 'y']].isna().any(axis=1)
        if missing.any():
            missing_ids = spot_aligned.index[missing].tolist()[:10]
            raise ValueError(
                f"spot 坐标与 MSI 强度矩阵无法按 Spot index 完全对齐，缺失示例: {missing_ids}"
            )

        x = spot_aligned['x'].astype(float).values
        y = spot_aligned['y'].astype(float).values
        obs_data = pd.DataFrame({
            'relative_x': (x - np.min(x)) / resolution,
            'relative_y': (y - np.min(y)) / resolution,
            'raw_x': x,
            'raw_y': y,
        }).reset_index(drop=True)

        X_data = np.nan_to_num(df_ms_t.values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        var_data = pd.DataFrame(df_ms_t.columns, index=df_ms_t.columns, columns=['m/z'])

    # ── 构建 AnnData ──
    adata = ad.AnnData(
        X=X_data,
        obs=obs_data,
        var=var_data,
        obsm={'spatial': obs_data[['relative_x', 'relative_y']].values}
    )
    # 存储空间分辨率，供配准坐标映射使用
    adata.uns['resolution'] = float(resolution)

    if save_path:
        adata.write(save_path)

    return adata


# ─── 归一化 ──────────────────────────────────────────────────

def normalize_adata(adata: ad.AnnData, method: str = 'TIC',
                    layer: str = None, inplace: bool = True) -> ad.AnnData:
    """
    对 AnnData 进行 TIC 或 RMS 归一化，原始 X 备份到 adata.uns['X_raw']。

    参数
    ----
    method  : 'TIC'（总离子流）或 'RMS'（均方根）
    inplace : False 则返回副本，不修改原对象
    """
    if not inplace:
        adata = adata.copy()

    # 备份原始数据
    if layer is None and 'X_raw' not in adata.uns:
        adata.uns['X_raw'] = adata.X.copy()
    elif layer is not None and f'{layer}_raw' not in adata.uns:
        adata.uns[f'{layer}_raw'] = adata.layers[layer].copy()

    X = np.nan_to_num(
        np.array(adata.X if layer is None else adata.layers[layer], dtype=np.float32),
        nan=0.0, posinf=0.0, neginf=0.0
    )

    if method.upper() == 'TIC':
        row_sums = X.sum(axis=1)
        row_sums[row_sums == 0] = 1
        X_norm = X / row_sums[:, None]
    elif method.upper() == 'RMS':
        rms = np.sqrt((X ** 2).sum(axis=1))
        rms[rms == 0] = 1
        X_norm = X / rms[:, None]
    else:
        raise ValueError("method 必须为 'TIC' 或 'RMS'")

    if layer is None:
        adata.X = X_norm
    else:
        adata.layers[layer] = X_norm

    return adata


# ─── 可视化（返回 base64，不弹 GUI）────────────────────────

def _fig_to_b64(fig: plt.Figure) -> str:
    """将 matplotlib Figure 转为 base64 PNG 字符串。"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64


def _msi_cmap():
    """实验记录同款 colormap：深蓝 → 蓝 → 青 → 绿 → 黄 → 橙 → 红。"""
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#00008B', '#0000FF', '#00FFFF', '#00FF00',
              '#FFFF00', '#FFA500', '#FF0000']
    return LinearSegmentedColormap.from_list('msi_cmap', colors)


def _spatial_scatter(ax, x, y, values, cmap, title, bg='#0f1117'):
    """
    在 ax 上绘制 MSI 空间热图（scatter），无坐标轴文字。

    参数
    ----
    x, y   : 像素坐标（relative_x / relative_y）
    values : 每像素的强度值（已 clip 到 99th percentile）
    """
    vmax = float(np.percentile(values, 99))
    vmin = 0.0
    sc = ax.scatter(x, y, c=values, cmap=cmap, s=1,
                    vmin=vmin, vmax=vmax, linewidths=0, rasterized=True)
    ax.set_facecolor('black')
    ax.set_aspect('equal')
    ax.invert_yaxis()          # 保持图像方向与扫描方向一致
    ax.set_axis_off()
    ax.set_title(title, color='#e2e8f0', fontsize=10, pad=4)
    cb = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cb.ax.tick_params(labelcolor='#e2e8f0', labelsize=7)
    # Use plain float format (avoid unreadable scientific notation for very small values)
    cb.formatter = matplotlib.ticker.FormatStrFormatter('%.2g')
    cb.update_ticks()


def plot_tic_histogram(adata_raw: ad.AnnData, adata_norm: ad.AnnData,
                       method: str = 'TIC') -> str:
    """
    生成归一化前后 MSI 空间热图对比（无中文标签）。

    左图：原始 TIC 空间分布（每像素总离子流）。
    右图：归一化校正因子（1/TIC 或 1/RMS），显示被校正的矩阵效应空间分布。
    使用蓝→红 colormap，与实验记录一致。
    返回 base64 编码的 PNG 字符串。
    """
    X_raw = np.array(adata_raw.uns.get('X_raw', adata_raw.X), dtype=np.float32)

    # 左图：原始 TIC
    tic_raw = np.nansum(X_raw, axis=1)

    # 右图：归一化校正因子（显示矩阵效应被校正的幅度）
    if method.upper() == 'TIC':
        denom = tic_raw.copy()
    else:  # RMS
        denom = np.sqrt(np.nansum(X_raw ** 2, axis=1))
    denom = np.where(denom > 0, denom, np.nan)
    norm_factor = 1.0 / denom      # 大值 = 该像素信号弱，被大幅放大

    x = adata_raw.obs['relative_x'].values
    y = adata_raw.obs['relative_y'].values

    cmap = _msi_cmap()
    BG   = '#0f1117'

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)
    _spatial_scatter(axes[0], x, y, tic_raw,     cmap, 'Raw TIC',                       BG)
    _spatial_scatter(axes[1], x, y, norm_factor, cmap, f'{method.upper()} correction factor', BG)

    plt.tight_layout(pad=1.5)
    return _fig_to_b64(fig)


def get_adata_stats(adata: ad.AnnData) -> dict:
    """返回 AnnData 基本统计信息（用于前端展示）。"""
    X = np.array(adata.X, dtype=np.float32)
    return {
        'n_pixels': int(adata.n_obs),
        'n_mz': int(adata.n_vars),
        'x_range': [float(adata.obs['relative_x'].min()), float(adata.obs['relative_x'].max())],
        'y_range': [float(adata.obs['relative_y'].min()), float(adata.obs['relative_y'].max())],
        'total_intensity_mean': float(X.sum(axis=1).mean()),
        'total_intensity_std': float(X.sum(axis=1).std()),
        'sparsity': float((X == 0).sum() / X.size),
    }

