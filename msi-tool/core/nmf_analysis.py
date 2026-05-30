"""
核心计算：NMF 空间积累模式分析
==================================
基于实验记录 12 实现：
  1. Moran's I 空间自相关筛选（不依赖 squidpy，内置实现）
  2. sklearn NMF 分解提取空间积累模式（topics）
  3. Spearman 秩相关识别各模式代表性代谢物
  4. 空间模式可视化（浅灰→深紫色图）+ top 代谢物空间分布图

算法说明
--------
Moran's I（Moran 1950; Cliff & Ord 1981）：
  - 行标准化 k-NN 权重矩阵：I = sum(z * lag_z) / sum(z^2)
  - 渐近正态近似方差：Var[I] ≈ 1/(N*k)（对行标准化 k-NN 的大样本近似）
  - 单侧检验（正向空间聚集）+ BH FDR 校正

NMF（Lee & Seung 1999）：
  - X_filtered (n_pixels × n_features) → W (n_pixels × n_comp), H (n_comp × n_features)
  - W 列为空间载荷（用于可视化），H 行为代谢物贡献权重
  - init='nndsvda' 保证稳定收敛

Spearman 相关（向量化）：
  - 对 X_filtered 和 W 的每列排秩，再计算 Pearson = Spearman
  - 结果：genepattern DataFrame (n_features × n_comp)

参考文献
--------
  Mirzal et al. (2022) IEEE/ACM Trans Comput Biol Bioinform 19(2):1173-1192
  Luo et al. (2023) Atherosclerosis 364:20-28
"""

import io
import base64
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = [
    'Microsoft YaHei', 'SimHei', 'SimSun',
    'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans',
]
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.decomposition import NMF
from scipy.spatial import cKDTree
from scipy.stats import norm, rankdata
import anndata as ad

warnings.filterwarnings('ignore')


# ──────────────────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────────────────

def _fig_to_b64(fig: plt.Figure) -> str:
    """matplotlib Figure → base64 PNG 字符串（关闭 figure 以释放内存）"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg FDR 校正。
    返回与输入等长的校正后 q 值数组（最大为 1.0）。
    """
    n = len(pvals)
    if n == 0:
        return np.array([])
    order = np.argsort(pvals)
    sorted_p = pvals[order]
    rank = np.arange(1, n + 1, dtype=float)
    adj = sorted_p * n / rank
    # 从右向左取运行最小值（确保单调非递减）
    for i in range(n - 2, -1, -1):
        adj[i] = min(adj[i], adj[i + 1])
    fdr = np.empty(n)
    fdr[order] = adj
    return np.minimum(fdr, 1.0)


def _make_nmf_cmap():
    """浅灰→中紫→深紫渐变色图（仿实验记录 12 seaborn blend_palette）"""
    return mcolors.LinearSegmentedColormap.from_list(
        'nmf_purple', ['#e8e8e8', '#b07cc6', '#8B008B']
    )


# ──────────────────────────────────────────────────────────────────────────────
# Moran's I 空间自相关计算
# ──────────────────────────────────────────────────────────────────────────────

def _compute_morans_i(X: np.ndarray, coords: np.ndarray,
                      n_neighbors: int = 10) -> pd.DataFrame:
    """
    对 X（n_pixels × n_features）中的每个特征计算 Moran's I 及正态近似 p 值。
    使用行标准化 k-NN 空间权重矩阵（不依赖 squidpy）。

    公式
    ----
    Moran's I = sum_i(z_i * lag_z_i) / sum_i(z_i^2)
    其中 z_i = x_i - mean(x)，lag_z_i = (1/k) * sum_{j in N(i)} z_j

    渐近正态近似（行标准化 k-NN，大样本近似）：
      E[I]   = -1/(N-1)
      Var[I] ≈ 1/(N * k)

    参数
    ----
    X          : ndarray (n_pixels, n_features)  归一化表达矩阵
    coords     : ndarray (n_pixels, 2)           空间坐标
    n_neighbors: int                             k-NN 邻居数

    返回
    ----
    DataFrame, columns=['I', 'pval_norm', 'pval_norm_fdr_bh']，index=range(n_features)
    """
    n_pixels, n_features = X.shape
    k = min(n_neighbors, n_pixels - 1)

    # ── 构建 k-NN 邻居索引 ────────────────────────────────────────────────
    tree = cKDTree(coords)
    _, knn_idx = tree.query(coords, k=k + 1)  # 第 0 列是自身
    neighbors = knn_idx[:, 1:]                  # (n_pixels, k)

    # ── 批量计算 Moran's I ────────────────────────────────────────────────
    morans_i = np.zeros(n_features, dtype=np.float64)
    for j in range(n_features):
        x = X[:, j].astype(np.float64)
        z = x - x.mean()
        denom = float(np.dot(z, z))
        if denom < 1e-12:
            morans_i[j] = 0.0
            continue
        # 空间滞后：每像素的 k 邻居 z 均值（行标准化权重）
        lag_z = z[neighbors].mean(axis=1)   # (n_pixels,)
        morans_i[j] = float(np.dot(z, lag_z)) / denom

    # ── 正态近似 p 值（单侧，检验正向空间聚集）────────────────────────────
    N = float(n_pixels)
    E_I  = -1.0 / (N - 1.0)
    Var_I = max(1.0 / (N * k), 1e-12)   # 行标准化 k-NN 大样本近似
    z_scores = (morans_i - E_I) / np.sqrt(Var_I)
    pvals    = 1.0 - norm.cdf(z_scores)  # one-sided
    fdr      = _bh_fdr(pvals)

    return pd.DataFrame({
        'I':                morans_i,
        'pval_norm':        pvals,
        'pval_norm_fdr_bh': fdr,
    })


# ──────────────────────────────────────────────────────────────────────────────
# NMF 分解
# ──────────────────────────────────────────────────────────────────────────────

def run_nmf(matrix: np.ndarray, n_components: int = 5,
            random_state: int = 16, max_iter: int = 2000) -> tuple:
    """
    sklearn NMF 分解。

    输入
    ----
    matrix      : (n_pixels, n_features)  非负表达矩阵
    n_components: int                     分量数（模式数）

    返回
    ----
    W                : (n_pixels, n_components)  空间载荷（每模式在像素的激活强度）
    H                : (n_components, n_features) 代谢物贡献权重
    reconstruction_err : float                   重构误差
    """
    mat = np.maximum(matrix, 0.0).astype(np.float64)

    # 与实验记录12 find_pattern() 保持一致：不加正则化（alpha=0）
    # TIC归一化后数据量级~0.001，alpha=0.05会使正则化梯度远大于重建梯度，
    # 导致W全部收敛到0，造成空间图全灰、Spearman ρ全零。
    try:
        model = NMF(
            n_components=n_components,
            init='nndsvda',
            random_state=random_state,
            max_iter=max_iter,
            alpha_W=0.0,
            alpha_H=0.0,
        )
        W = model.fit_transform(mat)
    except TypeError:
        model = NMF(
            n_components=n_components,
            init='nndsvda',
            random_state=random_state,
            max_iter=max_iter,
            alpha=0.0,
        )
        W = model.fit_transform(mat)

    return W, model.components_, model.reconstruction_err_


# ──────────────────────────────────────────────────────────────────────────────
# Spearman 相关（向量化）
# ──────────────────────────────────────────────────────────────────────────────

def _compute_spearman_matrix(X_filtered: np.ndarray,
                              W: np.ndarray,
                              feature_names) -> pd.DataFrame:
    """
    向量化计算代谢物与 NMF 模式载荷的 Spearman 秩相关矩阵。

    对每列排秩后，Pearson(ranked_X, ranked_W) = Spearman(X, W)。
    矩阵乘法加速：避免逐对调用 scipy.stats.spearmanr。

    返回
    ----
    DataFrame, shape (n_features, n_components)，
    columns = ['Pattern1', 'Pattern2', ...]
    """
    n_pixels, n_features = X_filtered.shape
    n_components = W.shape[1]

    def _rank_cols(M: np.ndarray) -> np.ndarray:
        ranked = np.empty_like(M, dtype=np.float64)
        for c in range(M.shape[1]):
            ranked[:, c] = rankdata(M[:, c])
        return ranked

    X_r = _rank_cols(X_filtered.astype(np.float64))  # (n_pixels, n_features)
    W_r = _rank_cols(W.astype(np.float64))            # (n_pixels, n_components)

    # 中心化
    X_z = X_r - X_r.mean(axis=0)   # (n_pixels, n_features)
    W_z = W_r - W_r.mean(axis=0)   # (n_pixels, n_components)

    X_norm = np.linalg.norm(X_z, axis=0) + 1e-12   # (n_features,)
    W_norm = np.linalg.norm(W_z, axis=0) + 1e-12   # (n_components,)

    # rho[j, i] = Spearman(X_filtered[:, j], W[:, i])
    rho = (X_z.T @ W_z) / (X_norm[:, np.newaxis] * W_norm[np.newaxis, :])
    # shape: (n_features, n_components)

    cols = [f'Pattern{i + 1}' for i in range(n_components)]
    return pd.DataFrame(rho.astype(np.float32), index=feature_names, columns=cols)


# ──────────────────────────────────────────────────────────────────────────────
# 可视化
# ──────────────────────────────────────────────────────────────────────────────

def plot_nmf_spatial_patterns(W: np.ndarray, coords: np.ndarray,
                               max_cutoff: float = 0.9) -> tuple:
    """
    生成所有 NMF 模式的空间载荷可视化。

    参数
    ----
    W          : (n_pixels, n_components)  NMF 空间载荷
    coords     : (n_pixels, 2)            空间坐标 [x, y]
    max_cutoff : float                    颜色上界分位数（超过部分截断）

    返回
    ----
    grid_b64       : str   合并网格图（base64 PNG）
    individual_b64 : list  每个模式单独图（list of base64 PNG）
    """
    n_components = W.shape[1]
    cmap = _make_nmf_cmap()
    x, y = coords[:, 0], coords[:, 1]

    # ── 合并网格图 ─────────────────────────────────────────────────────────
    n_cols = min(n_components, 5)
    n_rows = (n_components + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.2 * n_cols, 4.0 * n_rows),
                             facecolor='#121212')
    # 统一为二维数组
    if n_components == 1:
        axes_flat = [axes]
    elif n_rows == 1:
        axes_flat = list(axes)
    else:
        axes_flat = [ax for row in axes for ax in row]

    for i in range(n_components):
        ax = axes_flat[i]
        feature = W[:, i].copy()
        vmax = float(np.quantile(feature, max_cutoff))
        feature = np.clip(feature, 0.0, vmax)
        sc = ax.scatter(x, y, c=feature, cmap=cmap, s=2,
                        linewidths=0, vmin=0, vmax=max(vmax, 1e-10))
        ax.invert_yaxis()
        ax.set_title(f'Topic {i + 1}', color='white', fontsize=10, pad=4)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor('#121212')
        for sp in ax.spines.values():
            sp.set_visible(False)
        cb = plt.colorbar(sc, ax=ax, shrink=0.55, pad=0.02)
        cb.ax.yaxis.set_tick_params(color='white', labelcolor='white', labelsize=7)

    for i in range(n_components, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.tight_layout(pad=0.6)
    grid_b64 = _fig_to_b64(fig)

    # ── 每个模式单独图 ─────────────────────────────────────────────────────
    individual_b64 = []
    for i in range(n_components):
        fig_i, ax_i = plt.subplots(figsize=(4, 4), facecolor='#121212')
        feature = W[:, i].copy()
        vmax = float(np.quantile(feature, max_cutoff))
        feature = np.clip(feature, 0.0, vmax)
        sc = ax_i.scatter(x, y, c=feature, cmap=cmap, s=2,
                          linewidths=0, vmin=0, vmax=max(vmax, 1e-10))
        ax_i.invert_yaxis()
        ax_i.set_title(f'Topic {i + 1}', color='white', fontsize=12)
        ax_i.set_aspect('equal', adjustable='box')
        ax_i.set_xticks([]); ax_i.set_yticks([])
        ax_i.set_facecolor('#121212')
        for sp in ax_i.spines.values():
            sp.set_visible(False)
        cb = plt.colorbar(sc, ax=ax_i, shrink=0.6)
        cb.ax.yaxis.set_tick_params(color='white', labelcolor='white', labelsize=8)
        fig_i.tight_layout(pad=0.3)
        individual_b64.append(_fig_to_b64(fig_i))

    return grid_b64, individual_b64


def plot_top_metabolite_spatial(X_filtered: np.ndarray, coords: np.ndarray,
                                 feature_names, top_indices: list,
                                 pattern_idx: int,
                                 vmax_percentile: float = 99.0) -> str:
    """
    绘制某模式 top 代谢物的空间分布图（1 行 N 列）。
    返回 base64 PNG，色图使用 inferno（蓝→橙→白，突出高强度区域）。
    """
    n_top = len(top_indices)
    if n_top == 0:
        return ''

    cmap = plt.cm.inferno
    x, y = coords[:, 0], coords[:, 1]

    fig, axes = plt.subplots(1, n_top,
                             figsize=(3.5 * n_top, 3.5),
                             facecolor='#0f0f0f')
    if n_top == 1:
        axes = [axes]

    for k, idx in enumerate(top_indices):
        ax = axes[k]
        vals = X_filtered[:, idx].astype(float)
        vmax = float(np.percentile(vals, vmax_percentile))
        ax.scatter(x, y, c=vals, cmap=cmap, s=2,
                   linewidths=0, vmin=0, vmax=max(vmax, 1e-10))
        ax.invert_yaxis()
        label = str(feature_names[idx])
        if len(label) > 13:
            label = label[:12] + '…'
        ax.set_title(label, color='white', fontsize=8, pad=3)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor('#0f0f0f')
        for sp in ax.spines.values():
            sp.set_visible(False)

    fig.suptitle(f'Topic {pattern_idx + 1} — Top 代谢物空间分布',
                 color='white', fontsize=9, y=1.01)
    fig.tight_layout(pad=0.4)
    return _fig_to_b64(fig)


def get_top_metabolites(H: np.ndarray, mz_values, top_n: int = 10) -> list:
    """
    从 NMF H 矩阵（n_components × n_features）中提取每个模式权重最高的代谢物。
    （此函数基于 H 权重排序；主流程中改用 Spearman ρ 排序，生物学意义更强。）

    返回
    ----
    [{'component': i+1, 'metabolites': [{'mz': ..., 'weight': ...}]}]
    """
    result = []
    for i in range(H.shape[0]):
        w = H[i]
        top_idx = np.argsort(w)[::-1][:top_n]
        mets = [{'mz': str(mz_values[j]), 'weight': float(w[j])}
                for j in top_idx]
        result.append({'component': i + 1, 'metabolites': mets})
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────────────

def run_nmf_pipeline(h5ad_path: str,
                     n_components: int = 5,
                     cutoff: float = 0.5,
                     n_neighbors: int = 10,
                     top_n: int = 10,
                     moran_filter: bool = True,
                     max_cutoff: float = 0.9) -> dict:
    """
    完整空间 NMF 分析流程（实验记录 12 §3.2 对应实现）。

    步骤
    ----
    1. 加载 h5ad 数据（兼容稀疏矩阵）
    2. Moran's I 空间自相关筛选（可选）
    3. NMF 分解（sklearn，init=nndsvda）
    4. Spearman 相关矩阵计算（向量化）
    5. 空间模式可视化（合并图 + 单独图）
    6. 每模式 top 代谢物空间分布图

    参数
    ----
    h5ad_path    : str   h5ad 文件路径
    n_components : int   NMF 分量数（模式数），默认 5
    cutoff       : float Moran's I FDR 截断阈值（pval_norm_fdr_bh < cutoff），默认 0.5
    n_neighbors  : int   Moran's I 空间 k-NN 邻居数，默认 10
    top_n        : int   每模式展示的代表代谢物数（按 Spearman ρ 排序），默认 10
    moran_filter : bool  是否进行 Moran's I 筛选，默认 True
    max_cutoff   : float 空间图颜色上界分位数（超过部分截断），默认 0.9

    返回
    ----
    dict:
      pattern_grid_img  : str   合并网格图（base64 PNG）
      pattern_imgs      : list  各模式单独图（list of base64 PNG）
      top_metabolites   : list  每模式 top_n 代谢物（Spearman ρ 排序）
                                [{'pattern': i+1, 'rank': k, 'mz': ..., 'spearman_rho': ...}]
      top_met_imgs      : list  每模式 top 代谢物空间分布图（list of base64 PNG）
      spearman_table    : list  完整 Spearman 表（用于 CSV 导出）
      info              : dict  统计信息
    """
    # ── 1. 加载数据 ────────────────────────────────────────────────────────
    adata = ad.read_h5ad(h5ad_path)

    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = np.array(X, dtype=np.float32)
    n_pixels, n_features = X.shape

    # 空间坐标（优先使用 obsm['spatial']，回退到 obs 列）
    if 'spatial' in adata.obsm:
        coords = np.array(adata.obsm['spatial'], dtype=np.float64)
    elif 'relative_x' in adata.obs.columns and 'relative_y' in adata.obs.columns:
        coords = adata.obs[['relative_x', 'relative_y']].values.astype(np.float64)
    else:
        raise ValueError(
            '未找到空间坐标：请确保 adata.obsm["spatial"] 或 obs 中存在 relative_x/relative_y 列'
        )
    if coords.ndim == 1 or coords.shape[1] < 2:
        raise ValueError('空间坐标维度不正确，应为 (n_pixels, 2) 或更多列')
    coords = coords[:, :2]  # 只取前两列 (x, y)

    mz_values = np.array(adata.var.index.tolist())

    # ── 2. Moran's I 筛选 ─────────────────────────────────────────────────
    n_after_moran = n_features
    selected_idx  = np.arange(n_features)

    if moran_filter:
        moran_df = _compute_morans_i(X, coords, n_neighbors=n_neighbors)
        sig_mask    = moran_df['pval_norm_fdr_bh'].values < cutoff
        selected_idx = np.where(sig_mask)[0]
        n_after_moran = int(len(selected_idx))

        # 若筛选后特征数不足以进行 NMF，回退到 Moran's I 最大的特征
        min_needed = max(n_components * 3, 10)
        if n_after_moran < min_needed:
            fallback_k = max(n_components * 5, 30)
            selected_idx  = np.argsort(moran_df['I'].values)[::-1][:fallback_k]
            n_after_moran = int(len(selected_idx))

    X_filtered = X[:, selected_idx]          # (n_pixels, n_after_moran)
    mz_filtered = mz_values[selected_idx]    # (n_after_moran,)

    # ── 3. NMF 分解 ───────────────────────────────────────────────────────
    W, H, recon_err = run_nmf(X_filtered, n_components=n_components)
    # W: (n_pixels, n_components)  空间载荷
    # H: (n_components, n_after_moran)  代谢物权重

    # ── 4. Spearman 相关矩阵 ──────────────────────────────────────────────
    genepattern = _compute_spearman_matrix(X_filtered, W, mz_filtered)
    # DataFrame shape: (n_after_moran, n_components)

    # ── 5. 空间模式可视化 ──────────────────────────────────────────────────
    grid_b64, individual_b64 = plot_nmf_spatial_patterns(
        W, coords, max_cutoff=max_cutoff
    )

    # ── 6. 代表代谢物列表 + 空间分布图 ────────────────────────────────────
    top_metabolites_per_pattern = []
    top_met_imgs  = []
    spearman_table = []
    name_to_idx   = {str(name): idx for idx, name in enumerate(mz_filtered)}

    for i in range(n_components):
        col = f'Pattern{i + 1}'
        sorted_series = genepattern[col].sort_values(ascending=False)
        top_series    = sorted_series.head(top_n)

        mets = []
        for rank_j, (mz_name, rho) in enumerate(top_series.items(), start=1):
            entry = {
                'pattern':      i + 1,
                'rank':         rank_j,
                'mz':           str(mz_name),
                'spearman_rho': round(float(rho), 4),
            }
            mets.append(entry)
            spearman_table.append(entry)
        top_metabolites_per_pattern.append(mets)

        # top 代谢物在 X_filtered 中的列索引
        top_feat_idx = [
            name_to_idx[m['mz']]
            for m in mets
            if m['mz'] in name_to_idx
        ]
        if top_feat_idx:
            img = plot_top_metabolite_spatial(
                X_filtered, coords, mz_filtered, top_feat_idx, i
            )
        else:
            img = ''
        top_met_imgs.append(img)

    return {
        'pattern_grid_img':  grid_b64,
        'pattern_imgs':      individual_b64,
        'top_metabolites':   top_metabolites_per_pattern,
        'top_met_imgs':      top_met_imgs,
        'spearman_table':    spearman_table,
        'info': {
            'n_pixels':               int(n_pixels),
            'n_features_input':       int(n_features),
            'n_features_after_moran': int(n_after_moran),
            'n_components':           int(n_components),
            'moran_filter':           bool(moran_filter),
            'cutoff':                 float(cutoff),
            'reconstruction_err':     round(float(recon_err), 4),
        },
    }
