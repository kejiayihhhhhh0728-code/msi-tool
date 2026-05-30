"""
核心计算：降维与空间聚类（v4 算法）
====================================
输入: AnnData h5ad（经过 TIC/RMS 归一化）
输出: base64 PNG 图片 + 指标 dict

降维路线:
  路线 A (pca)     : PCA → 在 PCA 空间聚类
  路线 B (umap_dr) : PCA 预降维 → 多维 UMAP(10 维) → 在 UMAP 空间聚类

可视化 UMAP: 始终独立，从 PCA 邻域图生成 2D 嵌入，不参与聚类。

聚类方法（7 种）:
  KMeans / Agglomerative(Ward) / GMM / Spatial-Agg /
  Spectral / Leiden / HDBSCAN

参考文献:
  Laskin 2021 (PMC7904669)  — PCA+GMM, UMAP+HDBSCAN, 可视化规范
  DiviK 2022 (BMC Bioinf.)  — PCA+Ward aggl, Spectral 对比
  Traag 2019 / scanpy docs  — Leiden 算法
  umap-learn docs           — 聚类用 UMAP 参数建议
  Allaoui 2020 (LNCS)       — min_dist=0 利于下游聚类
"""

import io
import math
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
# ── 中文字体支持（Windows 优先 Microsoft YaHei，Linux 用文泉驿）──────────
matplotlib.rcParams['font.sans-serif'] = [
    'Microsoft YaHei', 'SimHei', 'SimSun',          # Windows
    'WenQuanYi Micro Hei', 'Noto Sans CJK SC',      # Linux
    'DejaVu Sans',                                   # 兜底
]
matplotlib.rcParams['axes.unicode_minus'] = False   # 负号不显示为方框
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import anndata as ad

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import kneighbors_graph
from scipy import spatial as sp_spatial
from scipy.ndimage import median_filter
import warnings
warnings.filterwarnings('ignore')


# ─── 工具 ─────────────────────────────────────────────────────────────

def _fig_to_b64(fig: plt.Figure) -> str:
    """Figure → base64 PNG"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64


def safe_silhouette(X, labels):
    if labels is None or len(np.unique(labels)) < 2:
        return float('nan')
    try:
        return float(silhouette_score(X, labels))
    except Exception:
        return float('nan')


def _make_discrete_colors(labels):
    unique = np.unique(labels)
    cmap = plt.cm.tab10 if len(unique) <= 10 else plt.cm.tab20
    lb_map = {lb: cmap(i % cmap.N) for i, lb in enumerate(unique)}
    pc = [lb_map[lb] for lb in labels]
    patches = [mpatches.Patch(color=cmap(i % cmap.N), label=f'C{lb}')
               for i, lb in enumerate(unique)]
    return pc, patches


METHOD_DISPLAY = {
    'kmeans':        'KMeans',
    'agglomerative': 'Agglomerative\n(Ward)',
    'gmm':           'GMM',
    'spatial':       'Spatial-Agg',
    'spectral':      'Spectral',
    'leiden':        'Leiden',
    'hdbscan':       'HDBSCAN',
}

BG = '#0f1117'  # 深色背景，与整体风格一致


# ─── 预处理 ───────────────────────────────────────────────────────────

def _preprocess(X: np.ndarray,
                zero_threshold: float = 0.8,
                variance_threshold: float = 0.01) -> tuple:
    """
    预处理流程：尺度还原 → log1p → 零值过滤 → 方差过滤 → RobustScaler

    X 是 TIC/RMS 归一化后的数据，行和约为 1，需先还原到计数量级再做 log1p。
    """
    X = np.nan_to_num(X.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    n_features = X.shape[1]

    row_sums = X.sum(axis=1)
    median_sum = float(np.median(row_sums[row_sums > 0]))
    if median_sum < 10.0:          # 已归一化（TIC≈1，RMS≈1）
        X = X * n_features         # 还原到计数量级

    X_log = np.log1p(X)

    zero_frac = (X == 0).mean(axis=0)
    mask0 = zero_frac < zero_threshold
    if mask0.sum() == 0:
        mask0 = np.ones(n_features, dtype=bool)
    X_f = X_log[:, mask0]

    var = X_f.var(axis=0)
    mask1 = var > variance_threshold
    if mask1.sum() == 0:
        mask1 = var > 0
    if mask1.sum() == 0:
        mask1 = np.ones(X_f.shape[1], dtype=bool)
    X_f = X_f[:, mask1]

    combined_mask = np.where(mask0)[0][mask1]

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_f)
    return X_scaled, combined_mask


def _spatial_smooth(X_scaled: np.ndarray, coords: np.ndarray,
                    radius: float = 2.0, alpha: float = 0.5) -> np.ndarray:
    """空间 KD-tree 加权平均平滑。"""
    if radius <= 0 or alpha <= 0:
        return X_scaled.copy()
    tree = sp_spatial.KDTree(coords)
    nbrs_list = tree.query_ball_tree(tree, r=radius)
    X_nbr = np.zeros_like(X_scaled)
    for i, nbrs in enumerate(nbrs_list):
        X_nbr[i] = np.mean(X_scaled[nbrs], axis=0) if len(nbrs) > 1 else X_scaled[i]
    return (1 - alpha) * X_scaled + alpha * X_nbr


# ─── 降维 ─────────────────────────────────────────────────────────────

def _reduce_pca(X_smoothed: np.ndarray,
                n_components: int = 30,
                variance_threshold: float = None) -> tuple:
    """
    PCA 降维。

    参数
    ----
    n_components      : 固定保留维度数（默认 30，参考 Laskin 2021）
    variance_threshold: 若设置（如 0.95），则改为保留累积方差达该比例的 PC 数，
                        忽略 n_components。
    """
    if variance_threshold is not None:
        n = variance_threshold          # PCA 接受 0<float<1 自动确定维度数
    else:
        n = min(n_components,
                X_smoothed.shape[0] - 1,
                X_smoothed.shape[1])
    pca = PCA(n_components=n, random_state=42)
    X_pca = pca.fit_transform(X_smoothed)
    return X_pca, pca


def _compute_umap_dr(X_pca: np.ndarray,
                     n_components: int = 10,
                     n_neighbors: int = 30,
                     min_dist: float = 0.0,
                     metric: str = 'cosine') -> np.ndarray:
    """
    多维 UMAP，用于聚类输入（路线 B）。

    参数来源:
      - n_components=10: umap-learn 文档，聚类目的推荐 5-15 维而非 2 维
      - n_neighbors=30 : 聚类用途建议比可视化更大（文档默认 15 偏小）
      - min_dist=0.0   : Allaoui 2020 建议设 0 使点更紧密，有利于下游聚类
      - metric='cosine': Laskin 2021 MSI 数据推荐 cosine 距离
    """
    try:
        import umap
        reducer = umap.UMAP(n_components=n_components,
                            n_neighbors=n_neighbors,
                            min_dist=min_dist,
                            metric=metric,
                            random_state=42)
        return reducer.fit_transform(X_pca)
    except ImportError:
        return None


def _compute_umap_vis(X_pca: np.ndarray,
                      n_neighbors: int = 15,
                      min_dist: float = 0.5) -> np.ndarray:
    """
    2D UMAP，仅用于可视化，始终从 PCA 空间计算，与聚类路线无关。

    参数来源:
      - min_dist=0.5: scanpy 默认值，适合展示全局结构
      - n_neighbors=15: scanpy 默认值
    """
    try:
        import umap
        reducer = umap.UMAP(n_components=2,
                            n_neighbors=n_neighbors,
                            min_dist=min_dist,
                            metric='euclidean',
                            random_state=42)
        return reducer.fit_transform(X_pca)
    except ImportError:
        return None


# ─── Leiden（可选依赖：leidenalg + igraph）──────────────────────────

def _try_leiden(X_pca: np.ndarray, target_k: int,
                n_neighbors: int = 15, max_iter: int = 60) -> np.ndarray | None:
    """
    二分搜索 resolution 使 Leiden 输出的簇数尽量接近 target_k。
    依赖: leidenalg, igraph（可选）。
    参考: Traag et al. 2019; scanpy PCA→neighbors→leiden 标准流程。
    """
    try:
        import leidenalg
        import igraph as ig
    except ImportError:
        return None

    A = kneighbors_graph(X_pca, n_neighbors=n_neighbors,
                         mode='connectivity', include_self=False)
    A_sym = A + A.T
    rows, cols = (A_sym > 0).nonzero()
    edges_dedup = list({(min(int(u), int(v)), max(int(u), int(v)))
                        for u, v in zip(rows, cols)})
    G = ig.Graph(n=X_pca.shape[0], edges=edges_dedup, directed=False)

    lo, hi = 0.01, 10.0
    best_labels, best_diff = None, float('inf')

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        part = leidenalg.find_partition(
            G, leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=mid, seed=42)
        lbs = np.array(part.membership)
        diff = abs(len(np.unique(lbs)) - target_k)
        if diff < best_diff:
            best_diff = diff
            best_labels = lbs.copy()
        if diff == 0:
            break
        elif len(np.unique(lbs)) < target_k:
            lo = mid
        else:
            hi = mid

    return best_labels


# ─── HDBSCAN（可选依赖）─────────────────────────────────────────────

def _try_hdbscan(X_dr: np.ndarray,
                 min_cluster_size: int = 100,
                 min_samples: int = 30) -> np.ndarray | None:
    """
    HDBSCAN 密度聚类，不需要预设 k，自动确定簇数并标记噪声点（-1）。

    参数来源: Laskin 2021 — min_cluster_size=300, min_samples=30（子宫数据）；
    默认值调小至 100 以适应更小的 MSI 数据集。

    优先使用独立 hdbscan 包；sklearn>=1.3 亦内置 HDBSCAN。
    在 UMAP 嵌入空间表现最佳（Laskin 2021 明确使用 UMAP+HDBSCAN 路线）。
    """
    try:
        import hdbscan as _hdbscan
        clusterer = _hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                     min_samples=min_samples,
                                     prediction_data=True)
        return clusterer.fit_predict(X_dr).astype(int)
    except ImportError:
        pass
    try:
        from sklearn.cluster import HDBSCAN as SkHDBSCAN
        return SkHDBSCAN(min_cluster_size=min_cluster_size,
                         min_samples=min_samples).fit_predict(X_dr).astype(int)
    except (ImportError, Exception):
        return None


# ─── 全方法聚类 ──────────────────────────────────────────────────────

SPECTRAL_LIMIT = 8000  # 像素数超过此值跳过 Spectral（affinity 矩阵内存过大）


def find_all_clustering(X_dr: np.ndarray,
                        coords: np.ndarray,
                        n_clusters: int = 5,
                        spatial_n_neighbors: int = 8,
                        gmm_auto_bic: bool = False,
                        gmm_search_range: int = 5,
                        hdbscan_min_cluster_size: int = 100,
                        hdbscan_min_samples: int = None,
                        X_hdbscan: np.ndarray = None) -> dict:
    """
    用 7 种方法在降维空间聚类，并列返回所有结果。

    参数
    ----
    X_dr         : 降维后的特征矩阵（PCA 或 UMAP-DR 空间），用于前 6 种方法
    coords       : 像素空间坐标，用于 Spatial-Agg
    n_clusters   : 目标簇数（HDBSCAN 自动确定，忽略此参数）
    X_hdbscan    : HDBSCAN 专用输入空间（应为低维，如 2D UMAP 或 UMAP-DR）；
                   为 None 时跳过 HDBSCAN。
                   参考: Laskin 2021 — HDBSCAN 需在低维嵌入空间中运行
    hdbscan_min_samples: None 时取 min_cluster_size（hdbscan 文档推荐默认值）

    返回
    ----
    dict: {method_name: {'labels': np.ndarray | None, 'silhouette': float,
                         'n_clusters_actual': int | None}}
    """
    k = n_clusters
    n = X_dr.shape[0]

    connectivity = kneighbors_graph(coords,
                                    n_neighbors=min(spatial_n_neighbors, n - 1),
                                    mode='connectivity', include_self=False)
    results = {}

    # ── KMeans ────────────────────────────────────────────────────────
    # 参考: SCiLS Lab, Talanta 2022, DiviK 2022（基准方法）
    lbs = KMeans(n_clusters=k, random_state=42, n_init=10,
                 max_iter=300).fit_predict(X_dr)
    results['kmeans'] = {'labels': lbs,
                         'silhouette': safe_silhouette(X_dr, lbs),
                         'n_clusters_actual': k}

    # ── Agglomerative (Ward) ──────────────────────────────────────────
    # 参考: DiviK 2022 — Ward linkage + Euclidean，保留 70% PCA 方差
    lbs = AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(X_dr)
    results['agglomerative'] = {'labels': lbs,
                                'silhouette': safe_silhouette(X_dr, lbs),
                                'n_clusters_actual': k}

    # ── GMM ───────────────────────────────────────────────────────────
    # 参考: Laskin 2021 — PCA+GMM；可选 BIC 扫描自动选 n_components
    try:
        if gmm_auto_bic:
            # BIC 扫描：在 [max(2, k-range), k+range] 内找最小 BIC
            lo_k = max(2, k - gmm_search_range)
            hi_k = k + gmm_search_range
            best_k, best_bic = k, float('inf')
            for nk in range(lo_k, hi_k + 1):
                try:
                    g = GaussianMixture(n_components=nk, random_state=42,
                                        covariance_type='full', reg_covar=1e-6,
                                        max_iter=200)
                    g.fit(X_dr)
                    bic = g.bic(X_dr)
                    if bic < best_bic:
                        best_bic, best_k = bic, nk
                except Exception:
                    pass
            lbs = GaussianMixture(n_components=best_k, random_state=42,
                                  covariance_type='full', reg_covar=1e-6,
                                  max_iter=200).fit_predict(X_dr)
            gmm_k = best_k
        else:
            lbs = GaussianMixture(n_components=k, random_state=42,
                                  covariance_type='full', reg_covar=1e-6,
                                  max_iter=200).fit_predict(X_dr)
            gmm_k = k
        results['gmm'] = {'labels': lbs,
                          'silhouette': safe_silhouette(X_dr, lbs),
                          'n_clusters_actual': gmm_k}
    except Exception:
        results['gmm'] = {'labels': None,
                          'silhouette': float('nan'),
                          'n_clusters_actual': None}

    # ── Spatial-constrained Agglomerative ─────────────────────────────
    # 参考: squidpy 空间邻域概念；connectivity 矩阵由像素坐标 KNN 图构建
    try:
        lbs = AgglomerativeClustering(n_clusters=k, linkage='ward',
                                      connectivity=connectivity).fit_predict(X_dr)
    except Exception:
        lbs = results['kmeans']['labels'].copy()
    results['spatial'] = {'labels': lbs,
                          'silhouette': safe_silhouette(X_dr, lbs),
                          'n_clusters_actual': k}

    # ── Spectral（大数据集跳过）──────────────────────────────────────
    # 参考: DiviK 2022 — 多数情况排名第二
    # assign_labels='kmeans'（设计文档推荐）
    if n <= SPECTRAL_LIMIT:
        try:
            lbs = SpectralClustering(n_clusters=k, random_state=42,
                                     n_neighbors=15,
                                     affinity='nearest_neighbors',
                                     assign_labels='kmeans').fit_predict(X_dr)
            results['spectral'] = {'labels': lbs,
                                   'silhouette': safe_silhouette(X_dr, lbs),
                                   'n_clusters_actual': k}
        except Exception:
            results['spectral'] = {'labels': None,
                                   'silhouette': float('nan'),
                                   'n_clusters_actual': None}
    else:
        results['spectral'] = {'labels': None,
                               'silhouette': float('nan'),
                               'n_clusters_actual': None}

    # ── Leiden（可选）────────────────────────────────────────────────
    # 参考: Traag 2019, scanpy PCA→neighbors→leiden 标准流程
    lbs = _try_leiden(X_dr, target_k=k)
    if lbs is not None:
        results['leiden'] = {'labels': lbs,
                             'silhouette': safe_silhouette(X_dr, lbs),
                             'n_clusters_actual': int(len(np.unique(lbs)))}
    else:
        results['leiden'] = {'labels': None,
                             'silhouette': float('nan'),
                             'n_clusters_actual': None}

    # ── HDBSCAN（可选）───────────────────────────────────────────────
    # 参考: Laskin 2021 — HDBSCAN 必须在低维嵌入空间（如 UMAP）中运行；
    #       hdbscan 文档 — min_samples=None 默认等于 min_cluster_size，
    #                      这是官方推荐的保守起点（McInnes et al. 2017, JOSS）
    if X_hdbscan is None:
        # 未提供低维空间，跳过（在 PCA 高维空间中运行会因维度诅咒产生大量噪声）
        results['hdbscan'] = {'labels': None,
                              'silhouette': float('nan'),
                              'n_clusters_actual': None}
    else:
        lbs = _try_hdbscan(X_hdbscan,
                           min_cluster_size=hdbscan_min_cluster_size,
                           min_samples=hdbscan_min_samples)
        if lbs is not None:
            n_actual = int(len(np.unique(lbs[lbs >= 0])))  # 排除噪声点 -1
            results['hdbscan'] = {
                'labels': lbs,
                'silhouette': safe_silhouette(
                    X_hdbscan[lbs >= 0], lbs[lbs >= 0])
                    if (lbs >= 0).sum() > 1 else float('nan'),
                'n_clusters_actual': n_actual,
            }
        else:
            results['hdbscan'] = {'labels': None,
                                  'silhouette': float('nan'),
                                  'n_clusters_actual': None}

    return results


# ─── 空间中值滤波 ─────────────────────────────────────────────────────

def spatial_median_filter(labels: np.ndarray, coords: np.ndarray,
                           kernel_size: int = 3) -> np.ndarray:
    """将聚类标签映射到 2D 网格，中值滤波后映射回来。"""
    if kernel_size <= 1:
        return labels.copy()

    x_vals, x_inv = np.unique(coords[:, 0], return_inverse=True)
    y_vals, y_inv = np.unique(coords[:, 1], return_inverse=True)
    h, w = len(y_vals), len(x_vals)

    flat = y_inv * w + x_inv
    cell_to_idx = {}
    for i, f in enumerate(flat):
        cell_to_idx.setdefault(int(f), []).append(i)

    grid = np.full((h, w), -1, dtype=np.int32)
    for f, idxs in cell_to_idx.items():
        yy, xx = divmod(f, w)
        lbs = labels[idxs].astype(int)
        mode_lb = np.bincount(lbs[lbs >= 0]).argmax() if (lbs >= 0).any() else 0
        grid[yy, xx] = int(mode_lb)

    filled = grid.copy()
    filled[grid < 0] = 0
    smoothed = median_filter(filled, size=kernel_size)

    result = labels.copy()
    for i in range(len(labels)):
        v = smoothed[y_inv[i], x_inv[i]]
        if v >= 0:
            result[i] = v
    return result


# ─── 可视化 ──────────────────────────────────────────────────────────

def _plot_grid(coords, X_vis, clustering_results, n_clusters,
               mode: str = 'spatial', dr_method: str = 'pca') -> str:
    """
    生成 N×3 网格图（mode='spatial' 或 mode='umap'）。
    有效方法排前面，跳过的方法排后面，空槽落在右下角。
    返回 base64 PNG。
    """
    # 有结果的方法排前面，跳过的排后面
    methods = sorted(clustering_results.keys(),
                     key=lambda m: clustering_results[m]['labels'] is None)

    ncols = 3
    nrows = math.ceil(len(methods) / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6 * ncols, 5 * nrows),
                             facecolor=BG)
    axes = axes.flatten() if nrows > 1 else np.array(axes).flatten()

    for idx, name in enumerate(methods):
        ax = axes[idx]
        ax.set_facecolor('black')
        val = clustering_results[name]
        lbs = val['labels']
        s = val['silhouette']
        k_actual = val.get('n_clusters_actual')

        if lbs is None:
            ax.text(0.5, 0.5, '跳过\n(依赖未安装 / 数据过大)',
                    ha='center', va='center', fontsize=11,
                    color='#64748b', transform=ax.transAxes)
        else:
            # HDBSCAN 噪声点（-1）用灰色显示，其余正常着色
            plot_lbs = lbs.copy()
            noise_mask = plot_lbs < 0
            plot_lbs[noise_mask] = plot_lbs[~noise_mask].max() + 1 \
                if (~noise_mask).any() else 0

            pc, patches = _make_discrete_colors(plot_lbs)
            # 噪声点强制灰色
            for i in np.where(noise_mask)[0]:
                pc[i] = (0.3, 0.3, 0.3, 0.5)

            if mode == 'spatial':
                ax.scatter(coords[:, 0], coords[:, 1],
                           c=pc, s=6, alpha=0.85, linewidths=0, rasterized=True)
                ax.invert_yaxis()
                ax.set_aspect('equal')
            else:
                if X_vis is None:
                    ax.text(0.5, 0.5, 'UMAP 可视化未启用',
                            ha='center', va='center', fontsize=11,
                            color='#64748b', transform=ax.transAxes)
                else:
                    ax.scatter(X_vis[:, 0], X_vis[:, 1],
                               c=pc, s=5, alpha=0.75, linewidths=0, rasterized=True)

            ax.legend(handles=patches, loc='lower right', fontsize=5.5,
                      framealpha=0.5, markerscale=1.3,
                      labelcolor='white', facecolor='#0f1117',
                      edgecolor='#2a2d3a')

        title = METHOD_DISPLAY.get(name, name)
        k_str = f'  k={k_actual}' if k_actual is not None else ''
        s_str = f'  sil={s:.3f}' if s == s else ''   # nan check
        ax.set_title(f'{title}{k_str}{s_str}',
                     color='#e2e8f0', fontsize=11, fontweight='bold', pad=6)
        ax.set_axis_off()

    for idx in range(len(methods), nrows * ncols):
        axes[idx].set_visible(False)

    dr_label = 'PCA' if dr_method == 'pca' else 'UMAP-DR'
    title_map = {
        'spatial': f'空间聚类分布（k={n_clusters}，降维: {dr_label}）',
        'umap':    f'UMAP 可视化嵌入（k={n_clusters}，降维: {dr_label}）',
    }
    fig.suptitle(title_map.get(mode, ''), color='#e2e8f0',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout(pad=0.8)
    return _fig_to_b64(fig)


# ─── 主入口 ──────────────────────────────────────────────────────────

def run_clustering_pipeline(
        h5ad_path: str,
        n_clusters: int = 5,
        # ── 降维路线 ──────────────────────────────────────────────────
        dr_method: str = 'pca',          # 'pca' 或 'umap_dr'
        pca_n_components: int = 30,      # 固定 PCA 维度（参考 Laskin 2021）
        pca_variance_threshold: float = None,  # 若设置则用累积方差代替固定维度
        umap_dr_n_components: int = 10,  # UMAP-DR 目标维度（umap-learn 文档建议 5-15）
        umap_dr_n_neighbors: int = 30,   # 聚类用途建议比可视化更大
        umap_dr_metric: str = 'cosine',  # Laskin 2021 推荐 cosine
        # ── 预处理 ────────────────────────────────────────────────────
        spatial_radius: float = 0.0,
        spatial_alpha: float = 0.0,
        median_kernel: int = 0,
        # ── 可视化 UMAP ───────────────────────────────────────────────
        do_umap_vis: bool = True,        # 是否生成 2D 可视化 UMAP
        # ── GMM ───────────────────────────────────────────────────────
        gmm_auto_bic: bool = False,      # 是否用 BIC 自动选 n_components
        gmm_search_range: int = 5,       # BIC 扫描范围 ±
        # ── HDBSCAN ───────────────────────────────────────────────────
        hdbscan_min_cluster_size: int = 100,
        hdbscan_min_samples: int = 30,
) -> dict:
    """
    从 h5ad 读取数据，执行完整聚类流程，返回图片和指标。

    返回 dict 包含:
      spatial_img : base64 PNG（空间分布多宫格）
      umap_img    : base64 PNG（UMAP 可视化嵌入，do_umap_vis=False 则为空字符串）
      metrics     : list[dict]（各方法指标）
      info        : dict（像素数、特征数、降维信息等）
      results_df  : pandas DataFrame（像素级聚类结果，供保存 CSV）
    """
    import pandas as pd

    # 1. 读取数据
    adata = ad.read_h5ad(h5ad_path)
    X = np.array(adata.X, dtype=np.float32)
    coords = adata.obs[['relative_x', 'relative_y']].values.astype(np.float32)
    n_pixels, n_mz_raw = X.shape
    if coords.shape[0] != n_pixels:
        raise ValueError(
            f"像素数与坐标数不一致: X={n_pixels}, coords={coords.shape[0]}。"
            "请检查 Spot index 对齐。")

    # 2. log1p + 过滤 + RobustScaler
    X_scaled, feat_mask = _preprocess(X)
    n_feat = X_scaled.shape[1]

    # 3. 空间平滑
    X_smoothed = _spatial_smooth(X_scaled, coords,
                                  radius=spatial_radius, alpha=spatial_alpha)

    # 4. PCA（所有路线都先做 PCA）
    X_pca, pca = _reduce_pca(X_smoothed,
                              n_components=pca_n_components,
                              variance_threshold=pca_variance_threshold)
    n_pca = X_pca.shape[1]

    # 5. 降维路线选择
    if dr_method == 'umap_dr':
        X_dr = _compute_umap_dr(X_pca,
                                n_components=umap_dr_n_components,
                                n_neighbors=umap_dr_n_neighbors,
                                metric=umap_dr_metric)
        if X_dr is None:
            # umap-learn 未安装，回退到 PCA
            X_dr = X_pca
            dr_method = 'pca'   # 更新标记，让图表标题正确
        n_dr = X_dr.shape[1]
    else:
        X_dr = X_pca
        n_dr = n_pca

    # 6. 可视化 UMAP（始终从 PCA 空间计算，独立于聚类路线）
    X_vis = _compute_umap_vis(X_pca) if do_umap_vis else None

    # 7. 确定 HDBSCAN 的输入空间
    # 路线 B (umap_dr)：X_dr 本身已是低维 UMAP 空间，直接使用
    # 路线 A (pca)    ：需要 2D 可视化 UMAP；若未启用则跳过 HDBSCAN
    # 参考: Laskin 2021 明确使用 UMAP 嵌入空间作为 HDBSCAN 输入
    if dr_method == 'umap_dr':
        X_hdbscan = X_dr           # UMAP-DR，低维，密度估计可靠
    elif X_vis is not None:
        X_hdbscan = X_vis          # 2D 可视化 UMAP，与 Laskin 2021 做法一致
    else:
        X_hdbscan = None           # 无可用低维空间，跳过

    # 8. 全方法聚类
    clustering_results = find_all_clustering(
        X_dr, coords,
        n_clusters=n_clusters,
        gmm_auto_bic=gmm_auto_bic,
        gmm_search_range=gmm_search_range,
        hdbscan_min_cluster_size=hdbscan_min_cluster_size,
        hdbscan_min_samples=hdbscan_min_samples,
        X_hdbscan=X_hdbscan,
    )

    # 9. 空间中值滤波
    if median_kernel > 0:
        for val in clustering_results.values():
            if val['labels'] is not None:
                val['labels'] = spatial_median_filter(
                    val['labels'], coords, median_kernel)
                val['silhouette'] = safe_silhouette(X_dr, val['labels'])

    # 10. 可视化
    spatial_img = _plot_grid(coords, X_vis, clustering_results,
                              n_clusters, mode='spatial', dr_method=dr_method)
    umap_img = _plot_grid(coords, X_vis, clustering_results,
                           n_clusters, mode='umap', dr_method=dr_method) \
               if do_umap_vis and X_vis is not None else ''

    # 11. 指标
    metrics = []
    for name, val in clustering_results.items():
        row = {
            'method': name,
            'display': METHOD_DISPLAY.get(name, name).replace('\n', ' '),
            'silhouette': round(val['silhouette'], 4)
                          if val['silhouette'] == val['silhouette'] else None,
            'n_clusters_actual': val.get('n_clusters_actual'),
            'available': val['labels'] is not None,
        }
        if val['labels'] is not None:
            lbs_valid = val['labels']
            try:
                row['ch_score'] = round(
                    float(calinski_harabasz_score(X_dr, lbs_valid)), 2)
                row['db_score'] = round(
                    float(davies_bouldin_score(X_dr, lbs_valid)), 4)
            except Exception:
                row['ch_score'] = None
                row['db_score'] = None
        metrics.append(row)

    # 12. 构建 DataFrame 供保存
    df = pd.DataFrame({
        'pixel_id': range(n_pixels),
        'x': coords[:, 0],
        'y': coords[:, 1],
    })
    if X_vis is not None:
        df['umap_vis_1'] = X_vis[:, 0]
        df['umap_vis_2'] = X_vis[:, 1]
    for name, val in clustering_results.items():
        if val['labels'] is not None:
            df[f'cluster_{name}'] = val['labels'].astype(int)

    return {
        'spatial_img': spatial_img,
        'umap_img':    umap_img,
        'metrics':     metrics,
        'results_df':  df,
        'info': {
            'n_pixels':               n_pixels,
            'n_mz_raw':               n_mz_raw,
            'n_features_after_filter': n_feat,
            'n_pca_dims':             n_pca,
            'n_dr_dims':              n_dr,
            'dr_method':              dr_method,
            'pca_variance_explained': round(
                float(pca.explained_variance_ratio_.sum()), 4),
        },
    }
