"""
核心计算：ROI 区域提取与 pseudo-bulk 聚合
不依赖 Flask，可单独测试

工作流程
--------
1. 用户在 HE 图（配准后叠加图）上绘制多边形 ROI
2. 利用配准仿射矩阵将每个 MSI 像素变换到 HE 坐标系
3. 判断变换后坐标是否落在多边形内
4. 提取匹配像素的强度矩阵 + 代谢物注释信息
5. 输出像素级宽表 CSV + pseudo-bulk 汇总
"""
from __future__ import annotations

import json
import os
import io
import base64
from typing import Optional

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from scipy.interpolate import RBFInterpolator


# ─── 工具函数 ───────────────────────────────────────────────────────

def load_affine_matrix(rigid_result_json: str) -> np.ndarray:
    """
    从 rigid_result.json 加载仿射矩阵，返回 3×3 numpy 数组。
    变换方向：[he_x, he_y, 1] = M @ [msi_x, msi_y, 1]
    """
    with open(rigid_result_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return np.array(data['affine_matrix'], dtype=np.float64)


def get_msi_raw_coords(adata: ad.AnnData) -> np.ndarray:
    """
    从 AnnData 获取用于仿射变换的 MSI 原始坐标 (n, 2)。

    优先使用 obs['raw_x'] / obs['raw_y']（由新版 preprocessing 存储）；
    若缺失则回退：relative_x * resolution（需 adata.uns['resolution']）。
    """
    obs = adata.obs
    if 'raw_x' in obs.columns and 'raw_y' in obs.columns:
        return obs[['raw_x', 'raw_y']].values.astype(np.float64)

    # 回退：用 relative_x * resolution 恢复近似原始坐标
    resolution = float(adata.uns.get('resolution', 1.0))
    rx = obs['relative_x'].values.astype(np.float64) * resolution
    ry = obs['relative_y'].values.astype(np.float64) * resolution
    return np.column_stack([rx, ry])


def msi_to_he_coords(msi_xy: np.ndarray, affine_matrix: np.ndarray) -> np.ndarray:
    """
    将 MSI 原始坐标变换到 HE 图像坐标系（仿射）。

    参数
    ----
    msi_xy        : (n, 2) MSI 原始坐标（与配准时相同的坐标系）
    affine_matrix : 3×3 仿射矩阵

    返回
    ----
    (n, 2) HE 图像坐标
    """
    n = len(msi_xy)
    ones = np.ones((n, 1))
    pts_h = np.hstack([msi_xy, ones])          # (n, 3)
    he_h  = (affine_matrix @ pts_h.T).T        # (n, 3)
    return he_h[:, :2]


def tps_forward_warp(
    points: np.ndarray,
    src_points: np.ndarray,
    dst_points: np.ndarray,
    smoothing: float = 0.0,
) -> np.ndarray:
    """
    Forward TPS warp：把 points (n, 2) 从 src 域映射到 dst 域。

    本工具的 src 是「affine-warped MSI 空间」中的控制点（用户在 tic_warped
    画布上的点击坐标），dst 是「HE 空间」对应控制点。给定任意 affine 已映射
    的 HE 坐标，本函数返回 TPS 精修后的 HE 坐标。

    参数
    ----
    points       : (n, 2) 待映射的点
    src_points   : (k, 2) 控制点对的 src 端，k >= 3
    dst_points   : (k, 2) 控制点对的 dst 端
    smoothing    : 0 = 精确插值（输出严格通过控制点）；
                   >0 时对噪声容忍度更高。默认 0。
    """
    src = np.asarray(src_points, dtype=np.float64)
    dst = np.asarray(dst_points, dtype=np.float64)
    if src.shape[0] < 3:
        raise ValueError('TPS forward warp 至少需要 3 对控制点')
    if src.shape != dst.shape:
        raise ValueError(f'src/dst 形状不一致: {src.shape} vs {dst.shape}')

    rbf = RBFInterpolator(src, dst, kernel='thin_plate_spline', smoothing=smoothing)
    return rbf(np.asarray(points, dtype=np.float64))


def compute_he_coords(
    adata: ad.AnnData,
    affine_matrix: np.ndarray,
    tps_src_points: Optional[list] = None,
    tps_dst_points: Optional[list] = None,
) -> np.ndarray:
    """
    把所有 MSI 像素一次性映射到 HE 坐标系。

    若提供 tps_src/dst_points（>= 3 对），在 affine 之后再做 forward TPS warp，
    返回 TPS 精修后的 HE 坐标；否则等价于 msi_to_he_coords。
    """
    msi_xy = get_msi_raw_coords(adata)
    he_xy = msi_to_he_coords(msi_xy, affine_matrix)
    if tps_src_points and tps_dst_points and len(tps_src_points) >= 3:
        he_xy = tps_forward_warp(
            he_xy,
            np.asarray(tps_src_points, dtype=np.float64),
            np.asarray(tps_dst_points, dtype=np.float64),
        )
    return he_xy


def pixels_in_roi(he_xy: np.ndarray, polygon_img: list) -> np.ndarray:
    """
    判断 HE 坐标系中的点是否落在多边形内。

    参数
    ----
    he_xy       : (n, 2) HE 图像坐标
    polygon_img : [[x1,y1], [x2,y2], ...] 已经换算到原图像素坐标的多边形顶点

    返回
    ----
    boolean array (n,)
    """
    path = MplPath(polygon_img)
    return path.contains_points(he_xy)


# ─── 主提取函数 ────────────────────────────────────────────────────

def extract_rois(
    adata: ad.AnnData,
    affine_matrix: np.ndarray,
    tissue_regions: Optional[list] = None,
    roi_list: Optional[list] = None,             # 旧接口，向后兼容
    tps_src_points: Optional[list] = None,
    tps_dst_points: Optional[list] = None,
) -> dict:
    """
    根据 HE 图上绘制的多边形 ROI 提取 MSI 像素数据。

    参数
    ----
    adata           : 含 obs[relative_x, relative_y] 及强度矩阵的 AnnData
    affine_matrix   : 3×3 仿射矩阵（MSI 原始坐标 → HE 图像坐标）
    tissue_regions  : 新接口（推荐）。两层结构：
                      [
                        {
                          'name'              : str,           组织区域名称（用户命名）
                          'boundary_polygons' : [[[x,y],...]] | None,
                                                                可选；仅做视觉/审计用，
                                                                不参与像素过滤
                          'sub_rois': [
                            {
                              'name'    : str,                  子区域名（如 '癌区1'）
                              'type'    : 'cancer'|'paracancer'|'custom',
                              'polygons': [[[x,y],...], ...]    一个或多个多边形，
                                                                取并集
                            },
                            ...
                          ]
                        },
                        ...
                      ]
    roi_list        : 旧接口（已废弃）。扁平 list:
                      [{'name', 'type', 'polygon_img'}, ...]
                      若提供，会被自动包装成单一无名组织，等效于新接口
    tps_src_points  : 可选，TPS 控制点的 src 端（affine-warped MSI 空间）
    tps_dst_points  : 可选，TPS 控制点的 dst 端（HE 空间）

    返回
    ----
    dict  {key: {
        'adata'         : subset AnnData,
        'pixel_count'   : int,
        'he_coords'     : np.ndarray (n,2)  HE 坐标（已含 TPS 修正，若启用）,
        'msi_obs'       : pd.DataFrame,
        'roi_type'      : str,
        'tissue_name'   : str,                 所属组织名（旧接口为空字符串）
        'sub_roi_name'  : str,
        'boundary_polygons': list | None,      该组织的外轮廓多边形（用于预览图）
    }}
    key 形如 '<tissue_name>__<sub_roi_name>'，组织名为空时为 '<sub_roi_name>'。
    """
    he_xy = compute_he_coords(adata, affine_matrix, tps_src_points, tps_dst_points)

    # 兼容旧接口：把 roi_list 包成单一无名组织
    if tissue_regions is None and roi_list:
        tissue_regions = [{
            'name': '',
            'boundary_polygons': None,
            'sub_rois': [
                {
                    'name'    : r['name'],
                    'type'    : r.get('type', 'custom'),
                    'polygons': [r['polygon_img']],
                }
                for r in roi_list
            ],
        }]
    if not tissue_regions:
        return {}

    results = {}
    for tissue in tissue_regions:
        tname = tissue.get('name', '') or ''
        boundary_polygons = tissue.get('boundary_polygons') or []
        tissue_mask = np.ones(len(he_xy), dtype=bool)
        if boundary_polygons:
            tissue_mask = np.zeros(len(he_xy), dtype=bool)
            for poly in boundary_polygons:
                if len(poly) >= 3:
                    tissue_mask |= pixels_in_roi(he_xy, poly)

        cancer_union = np.zeros(len(he_xy), dtype=bool)
        for sub in tissue.get('sub_rois', []):
            sname = sub.get('name', '') or ''
            polygons = sub.get('polygons', [])
            if not polygons:
                continue

            # 多 polygon 取并集（任一覆盖即算属于该子区域）
            inside = np.zeros(len(he_xy), dtype=bool)
            for poly in polygons:
                if len(poly) < 3:
                    continue
                inside |= pixels_in_roi(he_xy, poly)
            inside &= tissue_mask
            if sub.get('type') == 'cancer':
                cancer_union |= inside
            n_inside = int(inside.sum())
            if n_inside == 0:
                continue

            key = f'{tname}__{sname}' if tname else sname
            # 防止两个组织里有同名子区域时 key 冲突
            if key in results:
                key = f'{key}_dup{len(results)}'

            results[key] = {
                'adata'            : adata[inside].copy(),
                'pixel_count'      : n_inside,
                'he_coords'        : he_xy[inside],
                'msi_obs'          : adata.obs.iloc[inside].copy(),
                'roi_type'         : sub.get('type', 'custom'),
                'tissue_name'      : tname,
                'sub_roi_name'     : sname,
                'boundary_polygons': tissue.get('boundary_polygons'),
            }

        # 新交互逻辑：同一组织内，用户只画癌区；组织外轮廓内剩余像素自动归为癌旁。
        # 若用户仍手动提交了 paracancer，则不重复生成自动癌旁，保持向后兼容。
        has_manual_para = any(
            (sub.get('type') == 'paracancer') and sub.get('polygons')
            for sub in tissue.get('sub_rois', [])
        )
        if not has_manual_para and cancer_union.any():
            para_inside = tissue_mask & ~cancer_union
            n_para = int(para_inside.sum())
            if n_para > 0:
                sname = '自动癌旁'
                key = f'{tname}__{sname}' if tname else sname
                if key in results:
                    key = f'{key}_dup{len(results)}'
                results[key] = {
                    'adata'            : adata[para_inside].copy(),
                    'pixel_count'      : n_para,
                    'he_coords'        : he_xy[para_inside],
                    'msi_obs'          : adata.obs.iloc[para_inside].copy(),
                    'roi_type'         : 'paracancer',
                    'tissue_name'      : tname,
                    'sub_roi_name'     : sname,
                    'boundary_polygons': tissue.get('boundary_polygons'),
                }

    return results


# ─── 输出 CSV 构建 ────────────────────────────────────────────────

def build_intensity_dataframe(
    subset_adata: ad.AnnData,
    he_coords: Optional[np.ndarray] = None,
    annotation_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    构建像素级强度宽表 DataFrame。

    列顺序：relative_x, relative_y[, raw_x, raw_y][, he_x, he_y]
            + m/z列（若有注释则列名为 "m/z_value|代谢物名称"）

    参数
    ----
    subset_adata    : 某 ROI 的子 AnnData
    he_coords       : (n, 2) 该子集在 HE 坐标系中的位置（可选）
    annotation_df   : 代谢物注释表，需含 mz_observed / db_name 列（可选）
    """
    obs = subset_adata.obs.copy().reset_index(drop=True)

    # 提取强度矩阵
    X = subset_adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    mz_names = list(subset_adata.var_names)

    # 若有注释，构建 m/z → 代谢物名称的映射（取最高分命中）
    ann_map: dict = {}
    if annotation_df is not None and not annotation_df.empty:
        # 按 score_mass 降序，每个 m/z 取第一条记录
        ann_sorted = annotation_df.sort_values('score_mass', ascending=False)
        for _, row in ann_sorted.iterrows():
            try:
                mz_key = f"{float(row['mz_observed']):.4f}"
                if mz_key not in ann_map:
                    ann_map[mz_key] = str(row.get('db_name', ''))
            except Exception:
                pass

    # 构建 m/z 列名（带或不带代谢物注释）
    col_names = []
    for mz in mz_names:
        try:
            mz_key = f"{float(mz):.4f}"
        except Exception:
            mz_key = str(mz)
        name = ann_map.get(mz_key, '')
        col_names.append(f"{mz}|{name}" if name else str(mz))

    # 坐标列
    coord_cols = pd.DataFrame()
    for col in ['relative_x', 'relative_y', 'raw_x', 'raw_y']:
        if col in obs.columns:
            coord_cols[col] = obs[col].values

    if he_coords is not None:
        coord_cols['he_x'] = he_coords[:, 0]
        coord_cols['he_y'] = he_coords[:, 1]

    intensity_df = pd.DataFrame(X, columns=col_names)
    return pd.concat([coord_cols.reset_index(drop=True),
                      intensity_df.reset_index(drop=True)], axis=1)


def compute_pseudobulk(
    subset_adata: ad.AnnData,
    method: str = 'mean',
    annotation_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Pseudo-bulk 聚合：按均值（默认）或中位数压缩为单行向量。

    MSI 数据稀疏（大量像素强度为 0），median 容易把信号压成 0；与
    实验记录 13 的 compute_sample_means 一致，默认改为 mean。

    返回 DataFrame：1行，列为 m/z 名称（带注释列名）。
    """
    X = subset_adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    if method == 'median':
        agg = np.median(X, axis=0)
    elif method == 'mean':
        agg = np.mean(X, axis=0)
    else:
        raise ValueError(f'Unknown method: {method}')

    mz_names = list(subset_adata.var_names)

    # 注释映射（同 build_intensity_dataframe）
    ann_map: dict = {}
    if annotation_df is not None and not annotation_df.empty:
        ann_sorted = annotation_df.sort_values('score_mass', ascending=False)
        for _, row in ann_sorted.iterrows():
            try:
                mz_key = f"{float(row['mz_observed']):.4f}"
                if mz_key not in ann_map:
                    ann_map[mz_key] = str(row.get('db_name', ''))
            except Exception:
                pass

    col_names = []
    for mz in mz_names:
        try:
            mz_key = f"{float(mz):.4f}"
        except Exception:
            mz_key = str(mz)
        name = ann_map.get(mz_key, '')
        col_names.append(f"{mz}|{name}" if name else str(mz))

    return pd.DataFrame([agg], columns=col_names)


# ─── 预览图生成 ───────────────────────────────────────────────────

def plot_roi_preview(
    adata: ad.AnnData,
    affine_matrix: np.ndarray,
    roi_results: dict,
    he_img: Optional[np.ndarray] = None,
    tps_src_points: Optional[list] = None,
    tps_dst_points: Optional[list] = None,
) -> str:
    """
    生成 ROI 标注预览图，返回 base64 PNG。

    在 HE 空间中绘制各 ROI 区域像素（彩色散点）+ 多边形轮廓。
    若提供 he_img，则叠加在 HE 图像上显示。

    若提供 tps_src/dst_points，背景灰点同样会做 TPS forward warp，
    保证预览空间与提取结果一致（避免预览漂移）。
    """
    # 多组织调色板：(cancer_color, paracancer_color, custom_color)
    # 每组对应一种"组织"的色系。最多 3 个不同色系，第 4 个起回到第 1 个循环。
    TISSUE_PALETTES = [
        ('#f87171', '#34d399', '#38bdf8'),  # 组织 A：红 / 绿 / 蓝
        ('#a78bfa', '#60a5fa', '#fbbf24'),  # 组织 B：紫 / 蓝 / 黄
        ('#fb923c', '#22d3ee', '#a3e635'),  # 组织 C：橙 / 青 / 浅绿
    ]

    def color_for(tissue_idx: int, roi_type: str) -> str:
        palette = TISSUE_PALETTES[tissue_idx % len(TISSUE_PALETTES)]
        idx = {'cancer': 0, 'paracancer': 1, 'custom': 2}.get(roi_type, 2)
        return palette[idx]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8), facecolor='#0f1117')
    ax.set_facecolor('#0f1117')

    # 背景：全部 MSI 像素（灰色）—— 与 ROI 提取走同一映射通路
    he_all = compute_he_coords(adata, affine_matrix, tps_src_points, tps_dst_points)
    ax.scatter(he_all[:, 0], he_all[:, 1],
               s=1, c='#2a2d3a', alpha=0.5, rasterized=True)

    # 给每个 tissue_name 编号，决定其色系
    tissue_names = []
    for res in roi_results.values():
        tname = res.get('tissue_name', '') or ''
        if tname not in tissue_names:
            tissue_names.append(tname)

    # 各 ROI 区域着色
    for roi_key, res in roi_results.items():
        tname = res.get('tissue_name', '') or ''
        sname = res.get('sub_roi_name', '') or roi_key
        roi_type = res.get('roi_type', 'custom')
        t_idx = tissue_names.index(tname) if tname in tissue_names else 0
        color = color_for(t_idx, roi_type)
        coords = res['he_coords']
        label = f'{tname} / {sname}' if tname else sname
        ax.scatter(coords[:, 0], coords[:, 1],
                   s=2, c=color, alpha=0.8,
                   label=f'{label} ({res["pixel_count"]}px)',
                   rasterized=True)

    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title('ROI 提取预览（HE 坐标系）', color='#e2e8f0', fontsize=11)
    ax.tick_params(colors='#64748b')
    for spine in ax.spines.values():
        spine.set_edgecolor('#2a2d3a')
    if roi_results:
        legend = ax.legend(loc='upper right', fontsize=8,
                           facecolor='#1a1d27', labelcolor='#e2e8f0',
                           edgecolor='#2a2d3a')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()
