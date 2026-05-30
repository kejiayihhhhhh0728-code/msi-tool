"""
Step 1: 轮廓提取与几何属性计算
================================
提供 analyze_contour() 入口函数, 内部依次调用:
  1. extract_mask             — 图像 → 二值掩码 (含形态学清理)
  2. extract_largest_contour  — 取最大轮廓及面积
  3. compute_centroid         — 全 mask 前景像素均值质心
  4. resample_contour         — 等弧长重采样 (供 ICP 使用)
"""

import os
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image


# =====================================================================
# 3.1  图像读取与掩码提取
# =====================================================================

def extract_mask(input_file):
    """
    从图像文件提取二值掩码, 支持 RGBA / LA / L / RGB 格式。
    生成掩码后做形态学清理: 闭运算填锯齿缝隙, 开运算去噪点。

    Returns:
        mask: uint8 二值数组, shape (H, W), 值域 {0, 255}
    """
    im = Image.open(input_file)
    if im.mode == 'RGBA':
        arr = np.array(im)
        mask = (arr[:, :, 3] > 0).astype(np.uint8) * 255
    elif im.mode == 'LA':
        arr = np.array(im)
        mask = (arr[:, :, 1] > 0).astype(np.uint8) * 255
    elif im.mode == 'L':
        arr = np.array(im)
        mask = (arr > 0).astype(np.uint8) * 255
    else:
        gray = im.convert('L')
        arr = np.array(gray)
        mask = (arr > 0).astype(np.uint8) * 255
    # 形态学清理: 闭运算填锯齿缝隙, 开运算去噪点
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


# =====================================================================
# 3.2  最大轮廓提取
# =====================================================================

def extract_largest_contour(mask):
    """
    使用 cv2.findContours 检测外部轮廓, 选取面积最大的轮廓。

    Returns:
        cnt_pts: 轮廓点坐标 (N, 2), 或 None
        area:    轮廓面积 float
    """
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, 0.0
    cnt = max(contours, key=cv2.contourArea)
    cnt_pts = cnt.squeeze()
    if cnt_pts.ndim != 2 or len(cnt_pts) < 3:
        return None, 0.0
    area = cv2.contourArea(cnt)
    return cnt_pts, area


# =====================================================================
# 3.3  质心计算 (全 mask 前景像素均值)
# =====================================================================

def compute_centroid(mask):
    """
    找到所有前景像素 (值 > 0) 的坐标, 取 x 和 y 的均值作为质心。

    Returns:
        centroid: np.ndarray shape (2,), [x, y]
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.array([0.0, 0.0], dtype=np.float64)
    centroid = np.array([xs.mean(), ys.mean()], dtype=np.float64)
    return centroid


# =====================================================================
# 3.4  等弧长轮廓重采样
# =====================================================================

def resample_contour(pts, n_points=500):
    """
    按弧长等间距对闭合轮廓进行重采样。
    ICP 需要均匀分布的点才能正确匹配。

    Args:
        pts:      轮廓点集 (N, 2)
        n_points: 目标采样点数 (默认 500)

    Returns:
        重采样后的点集, shape (n_points, 2)
    """
    pts = np.array(pts, dtype=np.float64)
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0:1]])
    diffs = np.diff(pts, axis=0)
    seg_len = np.sqrt((diffs ** 2).sum(axis=1))
    arc = np.zeros(len(pts))
    arc[1:] = np.cumsum(seg_len)
    total = arc[-1]
    if total < 1e-6:
        return pts[:n_points] if len(pts) >= n_points else pts
    arc /= total  # 归一化到 [0, 1]
    t = np.linspace(0, 1, n_points, endpoint=False)
    x_interp = np.interp(t, arc, pts[:, 0])
    y_interp = np.interp(t, arc, pts[:, 1])
    return np.column_stack([x_interp, y_interp])


# =====================================================================
# 主入口: analyze_contour
# =====================================================================

def analyze_contour(image_path, output_dir=None, visualize=False,
                    n_sample=500):
    """
    从图像路径提取轮廓几何属性, 串联 extract_mask → extract_largest_contour
    → compute_centroid → resample_contour。

    Args:
        image_path: 图像文件路径 (RGBA/L PNG 均支持)
        output_dir: 可视化保存目录 (visualize=True 时使用)
        visualize:  是否保存轮廓可视化图
        n_sample:   轮廓重采样点数

    Returns:
        dict with keys:
            centroid        — (x, y) 全 mask 质心
            contour_area    — 最大轮廓面积
            contour_pts     — 原始轮廓点 (N, 2)
            contour_sampled — 等弧长重采样点 (n_sample, 2)
            n_pieces        — 连通组件数
            mask            — 二值掩码 uint8
        或 None (掩码为空时)
    """
    mask = extract_mask(image_path)

    # 连通组件数 (去背景)
    n_labels, _ = cv2.connectedComponents(mask)
    n_pieces = max(0, n_labels - 1)

    # 质心 (全 mask 前景像素均值)
    centroid = compute_centroid(mask)

    # 最大轮廓
    cnt_pts, area = extract_largest_contour(mask)
    if cnt_pts is None:
        print(f"  [step1] 警告: 未检测到有效轮廓 ({image_path})")
        return None

    # 等弧长重采样
    sampled = resample_contour(cnt_pts, n_points=n_sample)

    print(f"  [step1] {os.path.basename(image_path)}: "
          f"area={area:.0f}  n_pieces={n_pieces}  "
          f"centroid=({centroid[0]:.1f}, {centroid[1]:.1f})  "
          f"sampled={len(sampled)}")

    if visualize and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(mask, cmap='gray')
        ax.plot(cnt_pts[:, 0], cnt_pts[:, 1], 'r-', linewidth=1,
                label='contour')
        ax.plot(sampled[:, 0], sampled[:, 1], 'b.', markersize=2,
                label='sampled')
        ax.plot(centroid[0], centroid[1], 'g+', markersize=14,
                markeredgewidth=2, label='centroid')
        ax.set_title(
            f'area={area:.0f}  n_pieces={n_pieces}\n'
            f'centroid=({centroid[0]:.1f}, {centroid[1]:.1f})',
            fontsize=9)
        ax.legend(fontsize=8)
        ax.axis('off')
        base = os.path.splitext(os.path.basename(image_path))[0]
        fig.savefig(
            os.path.join(output_dir, f'{base}_contour.png'),
            dpi=150, bbox_inches='tight')
        plt.close(fig)

    return {
        'centroid': centroid,
        'contour_area': area,
        'contour_pts': cnt_pts,
        'contour_sampled': sampled,
        'n_pieces': n_pieces,
        'mask': mask,
    }
