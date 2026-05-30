"""
Step 2: 360° IoU+MI 联合搜索 + ICP 精细对齐
=============================================
配准流程:
  1. 缩放初始化: HE/MSI 轮廓面积比开方
  2. 质心对齐: MSI 全 mask 质心 → HE 全 mask 质心
  3. 360° 全搜索 (2° 步长): 每个角度算像素级 IoU
  4. 候选去重 (间距 < 6°)
  5. MI 联合打分: combined = IoU × 0.7 + MI × 0.3
  6. ICP 精细对齐: multi-start, 对 top 候选各跑 ICP, 选最佳

核心度量: 归一化互信息 (NMI), 能捕获 TIC 与 HE 灰度之间的非线性对应关系。
"""

import math
import numpy as np
import cv2
from scipy.spatial import cKDTree


# =====================================================================
# 3.5  几何工具函数
# =====================================================================

def build_affine_matrix(sx, sy, angle_rad, tx, ty, rot_center):
    """构建 3x3 仿射矩阵: 缩放 → 绕 rot_center 旋转 → 平移。"""
    cx, cy = rot_center
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)
    R = np.array([
        [c, -s, cx * (1 - c) + cy * s],
        [s,  c, cy * (1 - c) - cx * s],
        [0,  0, 1]
    ], dtype=np.float64)
    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
    return T @ R @ S


def apply_affine_to_points(pts, M):
    """将 3x3 仿射矩阵应用到 (N, 2) 点集。"""
    pts = np.array(pts, dtype=np.float64)
    ones = np.ones((len(pts), 1))
    homo = np.hstack([pts, ones])
    return (M @ homo.T).T[:, :2]


def warp_mask(src_mask, M_3x3, dst_shape):
    """用仿射矩阵变换二值掩码, 返回 float32。"""
    return cv2.warpAffine(
        src_mask.astype(np.float32),
        M_3x3[:2, :].astype(np.float64),
        (dst_shape[1], dst_shape[0]),
        flags=cv2.INTER_LINEAR,
        borderValue=0)


def pixel_iou(msi_mask, he_mask, M_3x3):
    """
    像素级 IoU。将 MSI 掩码通过仿射变换映射到 HE 坐标系, 在像素级计算 IoU。
    像素级 IoU 直接反映实际重叠面积, 不受凸包近似误差影响。

    Returns: (iou, overlap, union)
    """
    warped = warp_mask(msi_mask, M_3x3, he_mask.shape)
    w_bin = warped > 0.5
    h_bin = he_mask > 0.5
    overlap = int(np.sum(w_bin & h_bin))
    union = int(np.sum(w_bin | h_bin))
    iou = overlap / union if union > 0 else 0.0
    return iou, overlap, union


# =====================================================================
# 3.6  TIC 图像与 HE 灰度图构建
# =====================================================================

def create_tic_image(msi_df, msi_shape):
    """
    从 MSI DataFrame 构建 TIC (Total Ion Current) 图像。
    每个像素的值 = 该位置所有 m/z 通道强度之和。
    TIC 图像反映组织各区域的整体离子信号强度分布。
    """
    meta_cols = {'Spot index', 'x', 'y'}
    mz_cols = [c for c in msi_df.columns if c not in meta_cols]
    tic = np.zeros(msi_shape, dtype=np.float64)
    xs = msi_df['x'].values.astype(int)
    ys = msi_df['y'].values.astype(int)
    intensities = msi_df[mz_cols].values.astype(np.float64).sum(axis=1)
    for i in range(len(xs)):
        if 0 <= ys[i] < msi_shape[0] and 0 <= xs[i] < msi_shape[1]:
            tic[ys[i], xs[i]] = intensities[i]
    return tic


def he_to_grayscale(he_image_path):
    """
    加载 HE 图像, alpha 预乘混合到白色背景后,
    按 ITU-R BT.601 标准权重转换为灰度图。
    """
    from PIL import Image
    rgba = np.array(Image.open(he_image_path).convert('RGBA'))
    alpha = rgba[:, :, 3].astype(np.float32) / 255.0
    rgb = rgba[:, :, :3].astype(np.float32)
    for c in range(3):
        rgb[:, :, c] = rgb[:, :, c] * alpha + 255.0 * (1.0 - alpha)
    gray = 0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]
    return gray.astype(np.float64)


# =====================================================================
# 3.7  归一化互信息 (NMI) 计算
# =====================================================================

def _compute_mi(tic_norm, he_norm, msi_mask_bin, he_mask_bin, M_3x3,
                n_bins=64):
    """
    在重叠区域计算归一化互信息 (Normalized Mutual Information, NMI)。

    NMI = (H(A) + H(B)) / H(A, B)
      - 完全独立: NMI = 1; 完全相关: NMI = 2
      - 归一化到 [0, 1]: score = NMI - 1

    对齐越好, TIC 与 HE 灰度之间的统计依赖越强, NMI 越大。

    Args:
        tic_norm:     归一化 TIC 图像 [0, 1]
        he_norm:      归一化 HE 灰度图 [0, 1]
        msi_mask_bin: MSI 二值掩码
        he_mask_bin:  HE 二值掩码
        M_3x3:        3x3 仿射矩阵
        n_bins:       直方图 bin 数 (默认 64)

    Returns:
        float: NMI - 1, 范围 [0, 1]
    """
    warped_tic = cv2.warpAffine(
        tic_norm.astype(np.float32),
        M_3x3[:2, :].astype(np.float64),
        (he_norm.shape[1], he_norm.shape[0]),
        flags=cv2.INTER_LINEAR, borderValue=0)
    warped_mask = cv2.warpAffine(
        msi_mask_bin.astype(np.float32),
        M_3x3[:2, :].astype(np.float64),
        (he_norm.shape[1], he_norm.shape[0]),
        flags=cv2.INTER_LINEAR, borderValue=0)
    # 仅在重叠区域计算
    overlap = (warped_mask > 0.5) & (he_mask_bin > 0.5)
    n = np.sum(overlap)
    if n < 100:
        return 0.0

    a = warped_tic[overlap]
    b = he_norm[overlap]

    # 量化到 [0, n_bins-1]
    a_q = np.clip((a * (n_bins - 1)).astype(np.int32), 0, n_bins - 1)
    b_q = np.clip((b * (n_bins - 1)).astype(np.int32), 0, n_bins - 1)

    # 联合直方图 (bincount 向量化)
    flat_idx = a_q * n_bins + b_q
    joint_hist = np.bincount(
        flat_idx, minlength=n_bins * n_bins
    ).astype(np.float64).reshape(n_bins, n_bins)

    # 归一化为概率
    joint_prob = joint_hist / joint_hist.sum()

    # 边缘概率
    p_a = joint_prob.sum(axis=1)
    p_b = joint_prob.sum(axis=0)

    # 熵计算
    eps = 1e-12
    H_a = -np.sum(p_a[p_a > eps] * np.log(p_a[p_a > eps]))
    H_b = -np.sum(p_b[p_b > eps] * np.log(p_b[p_b > eps]))
    H_ab = -np.sum(joint_prob[joint_prob > eps] * np.log(joint_prob[joint_prob > eps]))

    if H_ab < eps:
        return 0.0

    # NMI = (H(A) + H(B)) / H(A,B), 范围 [1, 2]
    nmi = (H_a + H_b) / H_ab
    return max(0.0, float(nmi - 1.0))


# =====================================================================
# 角度搜索辅助
# =====================================================================

def _downsample_mask(mask, scale):
    if scale >= 0.999:
        return mask.astype(np.float32), 1.0
    new_w = max(1, int(round(mask.shape[1] * scale)))
    new_h = max(1, int(round(mask.shape[0] * scale)))
    resized = cv2.resize(mask.astype(np.float32), (new_w, new_h),
                         interpolation=cv2.INTER_NEAREST)
    return resized, scale


def _prepare_search_masks(msi_mask_bin, he_mask_bin, max_side=1600):
    max_dim = max(he_mask_bin.shape)
    scale = min(1.0, float(max_side) / float(max_dim))
    msi_search, _ = _downsample_mask(msi_mask_bin, scale)
    he_search, _ = _downsample_mask(he_mask_bin, scale)
    return msi_search, he_search, scale


def _build_M_at_angle(iso_scale, angle_deg, msi_centroid, he_centroid):
    M = build_affine_matrix(
        iso_scale, iso_scale, math.radians(angle_deg),
        tx=0, ty=0, rot_center=msi_centroid)
    lc = apply_affine_to_points([msi_centroid], M)[0]
    M[:2, 2] += np.array([he_centroid[0] - lc[0], he_centroid[1] - lc[1]])
    return M


# =====================================================================
# 3.8  360° 全搜索 + MI 联合打分
# =====================================================================

def register_iou_mi(msi_geo, he_geo, msi_mask_bin, he_mask_bin,
                    tic_image, he_gray,
                    step_deg=2, n_candidates=5, mi_weight=0.3,
                    n_bins=64):
    """
    360° 全搜索 + IoU & MI 联合打分。

    流程:
      第 1 步: 缩放初始化 + 360° IoU 全搜索 (2° 步长, 共 180 次评估)
      第 2 步: 候选去重 (间距 < 6°)
      第 3 步: MI 联合打分: combined = IoU × (1-w) + MI × w

    Args:
        msi_geo, he_geo: step1_geometric_analysis.analyze_contour 的返回结果
        msi_mask_bin, he_mask_bin: 二值掩码 (float32)
        tic_image: TIC 图像 (float64)
        he_gray:   HE 灰度图 (float64)
        step_deg:  角度搜索步长 (度)
        n_candidates: 返回候选数
        mi_weight: MI 在联合打分中的权重 (0~1)
        n_bins:    MI 直方图 bin 数

    Returns:
        best_M: 3x3 仿射矩阵
        results: dict
    """
    msi_centroid = msi_geo['centroid']
    he_centroid = he_geo['centroid']

    # 缩放: 面积比开方
    iso_scale = math.sqrt(
        he_geo['contour_area'] / msi_geo['contour_area']
    ) if msi_geo['contour_area'] > 0 else 1.0

    print(f"\n  [配准] 缩放系数: {iso_scale:.4f}")
    print(f"    MSI 质心: ({msi_centroid[0]:.1f}, {msi_centroid[1]:.1f})  "
          f"组织片数: {msi_geo.get('n_pieces', '?')}")
    print(f"    HE  质心: ({he_centroid[0]:.1f}, {he_centroid[1]:.1f})  "
          f"组织片数: {he_geo.get('n_pieces', '?')}")

    # --- 第 1 步: 360° 全搜索 ---
    msi_search_mask, he_search_mask, search_scale = _prepare_search_masks(
        msi_mask_bin, he_mask_bin)
    search_msi_centroid = (msi_centroid[0] * search_scale,
                           msi_centroid[1] * search_scale)
    search_he_centroid = (he_centroid[0] * search_scale,
                          he_centroid[1] * search_scale)

    print(f"\n  [配准] 360° 全搜索 ({step_deg}° 步长, "
          f"{int(360 / step_deg)} 次 IoU 计算)...")
    if search_scale < 0.999:
        print(f"    粗搜索降采样: scale={search_scale:.4f}  "
              f"HE {he_mask_bin.shape} -> {he_search_mask.shape}")

    search_angles = np.arange(0, 360, step_deg)
    angle_iou = []
    for a in search_angles:
        M_test = _build_M_at_angle(
            iso_scale, a, search_msi_centroid, search_he_centroid)
        iou, _, _ = pixel_iou(msi_search_mask, he_search_mask, M_test)
        angle_iou.append((a, iou))

    angle_iou.sort(key=lambda x: -x[1])

    print("    IoU top-5:")
    for a, iou in angle_iou[:5]:
        print(f"      {a:5.1f}°: IoU={iou:.4f}")

    # --- 第 2 步: 候选去重 (间距 < 6°) ---
    clustered = []
    used_angles = []
    for angle, iou in angle_iou:
        is_dup = False
        for ua in used_angles:
            diff = abs(angle - ua)
            diff = min(diff, 360 - diff)
            if diff < 6:
                is_dup = True
                break
        if not is_dup:
            clustered.append((angle, iou))
            used_angles.append(angle)

    print(f"    去重后候选 ({len(clustered)} 个, 间距≥6°):")
    for a, iou in clustered[:8]:
        print(f"      {a:6.2f}°: IoU={iou:.4f}")

    # --- 第 3 步: MI 联合打分 ---
    has_mi = tic_image is not None and he_gray is not None
    if has_mi:
        print(f"\n  [配准] MI 联合打分 "
              f"(权重: IoU={1-mi_weight:.1f}, MI={mi_weight:.1f}, bins={n_bins})...")

        tic_norm = tic_image.copy().astype(np.float64)
        if tic_norm.max() > 0:
            tic_norm /= tic_norm.max()
        he_norm = he_gray.copy().astype(np.float64)
        if he_norm.max() > 0:
            he_norm /= he_norm.max()

        candidates_scored = []
        for angle, iou in clustered:
            M = _build_M_at_angle(iso_scale, angle, msi_centroid, he_centroid)
            mi = _compute_mi(tic_norm, he_norm, msi_mask_bin, he_mask_bin, M,
                             n_bins=n_bins)
            combined = iou * (1 - mi_weight) + mi * mi_weight
            candidates_scored.append((angle, iou, mi, combined, M))

        candidates_scored.sort(key=lambda x: -x[3])

        print("    联合打分 top-5:")
        for a, iou, mi, comb, _ in candidates_scored[:5]:
            print(f"      {a:6.2f}°: IoU={iou:.4f}  MI={mi:.4f}  "
                  f"combined={comb:.4f}")
    else:
        print("\n  [配准] 无 TIC/HE 灰度, 仅用 IoU 打分")
        candidates_scored = []
        for angle, iou in clustered:
            M = _build_M_at_angle(iso_scale, angle, msi_centroid, he_centroid)
            candidates_scored.append((angle, iou, 0.0, iou, M))

    # 取 top-N
    top_n = candidates_scored[:n_candidates]
    candidates = [(item[4], item[1], item[0]) for item in top_n]

    best_M, best_iou, best_angle = candidates[0]
    print(f"\n  [配准] 最佳: {best_angle:.2f}° IoU={best_iou:.4f}")

    return best_M, {
        'method': 'IoU+MI',
        'iso_scale': iso_scale,
        'rotation_angle': best_angle,
        'best_iou': best_iou,
        'candidates': candidates,
    }


# =====================================================================
# 3.9  ICP 精细对齐
# =====================================================================

def icp_refine(src_pts, dst_pts, init_M=None,
               max_iter=50, tolerance=0.01, max_ratio=2.5):
    """
    ICP 精细对齐 (4 DoF: 均匀缩放 + 旋转 + 平移)。

    对 HE 采样点建立 KDTree, 查询变换后 MSI 点的最近邻。
    以中位数距离 × max_ratio 为阈值过滤异常值。
    使用 cv2.estimateAffinePartial2D + RANSAC 估计增量变换。
    当平均距离变化量 < tolerance 时收敛。
    """
    src = np.array(src_pts, dtype=np.float64)
    dst = np.array(dst_pts, dtype=np.float64)
    M = init_M.copy() if init_M is not None else np.eye(3, dtype=np.float64)
    tree = cKDTree(dst)
    prev_mean_dist = float('inf')

    for i in range(max_iter):
        transformed = apply_affine_to_points(src, M)
        distances, indices = tree.query(transformed)

        median_dist = np.median(distances)
        valid = distances < median_dist * max_ratio
        if valid.sum() < 4:
            break

        src_valid = transformed[valid].astype(np.float32)
        dst_valid = dst[indices[valid]].astype(np.float32)

        M_partial, _ = cv2.estimateAffinePartial2D(
            src_valid.reshape(-1, 1, 2),
            dst_valid.reshape(-1, 1, 2),
            method=cv2.RANSAC,
            ransacReprojThreshold=median_dist * 2)

        if M_partial is None:
            break

        M_partial_3x3 = np.eye(3, dtype=np.float64)
        M_partial_3x3[:2, :] = M_partial
        M = M_partial_3x3 @ M

        transformed_new = apply_affine_to_points(src, M)
        dists_new, _ = tree.query(transformed_new)
        mean_dist = np.mean(dists_new)

        if abs(prev_mean_dist - mean_dist) < tolerance:
            break
        prev_mean_dist = mean_dist

    final_transformed = apply_affine_to_points(src, M)
    final_dists, _ = tree.query(final_transformed)
    final_mean = np.mean(final_dists)

    return M, final_mean


def _extract_rotation_deg(M):
    return math.degrees(math.atan2(M[1, 0], M[0, 0]))


def _extract_scale(M):
    sx = math.sqrt(M[0, 0]**2 + M[1, 0]**2)
    sy = math.sqrt(M[0, 1]**2 + M[1, 1]**2)
    return (sx + sy) / 2.0


def run_icp(msi_geo, he_geo, M_init, msi_mask_bin, he_mask_bin,
            icp_max_iter=50, icp_tolerance=0.01,
            max_rotation_change=3.0, max_scale_change=0.05):
    """
    在轮廓采样点上运行 ICP, 带安全网:
      - IoU 不下降
      - 旋转变化 ≤ max_rotation_change°
      - 缩放变化 ≤ max_scale_change
    任一条件不满足则回退到 M_init。
    """
    print(f"\n  [ICP] 轮廓采样点精细对齐 (max_iter={icp_max_iter})...")
    print(f"    安全网: 旋转变化≤{max_rotation_change:.1f}°, "
          f"缩放变化≤{max_scale_change*100:.0f}%")

    msi_sampled = msi_geo['contour_sampled']
    he_sampled = he_geo['contour_sampled']
    print(f"    MSI 采样点: {len(msi_sampled)}  HE 采样点: {len(he_sampled)}")

    iou_before, _, _ = pixel_iou(msi_mask_bin, he_mask_bin, M_init)
    rot_before = _extract_rotation_deg(M_init)
    scale_before = _extract_scale(M_init)
    print(f"    ICP 前 IoU: {iou_before:.4f}  旋转: {rot_before:.2f}°  "
          f"缩放: {scale_before:.4f}")

    M_icp, mean_dist = icp_refine(
        msi_sampled, he_sampled,
        init_M=M_init,
        max_iter=icp_max_iter,
        tolerance=icp_tolerance)

    iou_after, _, _ = pixel_iou(msi_mask_bin, he_mask_bin, M_icp)
    rot_after = _extract_rotation_deg(M_icp)
    scale_after = _extract_scale(M_icp)

    rot_change = abs(rot_after - rot_before)
    rot_change = min(rot_change, 360 - rot_change)
    scale_ratio = abs(scale_after / scale_before - 1.0) if scale_before > 0 else 0.0
    iou_gain = iou_after - iou_before

    print(f"    ICP 后 IoU: {iou_after:.4f}  旋转: {rot_after:.2f}°  "
          f"缩放: {scale_after:.4f}  平均距离: {mean_dist:.2f}px")
    print(f"    Δ IoU: {iou_gain:+.4f}  Δ旋转: {rot_change:.2f}°  "
          f"Δ缩放: {scale_ratio*100:.2f}%")

    adopt = (iou_gain >= 0
             and rot_change <= max_rotation_change
             and scale_ratio <= max_scale_change)

    if adopt:
        print(f"    → 采用 ICP 结果")
        return M_icp, {
            'iou_before': iou_before, 'iou_after': iou_after,
            'mean_dist': mean_dist, 'rot_change': rot_change,
            'scale_change': scale_ratio, 'adopted': True,
        }
    else:
        reasons = []
        if iou_gain < 0:
            reasons.append(f"IoU 下降 ({iou_gain:+.4f})")
        if rot_change > max_rotation_change:
            reasons.append(f"旋转变化过大 ({rot_change:.2f}° > {max_rotation_change}°)")
        if scale_ratio > max_scale_change:
            reasons.append(f"缩放变化过大 ({scale_ratio*100:.2f}% > {max_scale_change*100:.0f}%)")
        print(f"    → 回退: {'; '.join(reasons)}")
        return M_init, {
            'iou_before': iou_before, 'iou_after': iou_after,
            'mean_dist': mean_dist, 'rot_change': rot_change,
            'scale_change': scale_ratio, 'adopted': False,
        }
