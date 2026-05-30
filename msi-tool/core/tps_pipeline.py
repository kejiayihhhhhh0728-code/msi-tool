import csv
import os
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image

from core.reg_step3 import fit_tps, build_tps_remap
from core.reg_pipeline import combine_masks_overlay

TPS_REG = 1000


def _calc_iou(a: np.ndarray, b: np.ndarray) -> float:
    ab = (a > 0.5) & (b > 0.5)
    un = (a > 0.5) | (b > 0.5)
    return float(np.sum(ab) / np.sum(un)) if np.sum(un) > 0 else 0.0


def run_tps_registration(session_dir: str, result_dir: str,
                         pairs: List[Dict], affine_matrix) -> Dict:
    """
    pairs: [{"src": [x, y], "dst": [x, y]}, ...]
    依赖 session_dir 中 reg_pipeline.py 生成的 msi_mask_original.png 和 he_mask.png。
    """
    os.makedirs(result_dir, exist_ok=True)

    if len(pairs) < 3:
        raise ValueError('TPS requires at least 3 landmark pairs.')

    msi_mask_path = os.path.join(session_dir, 'msi_mask_original.png')
    he_mask_path  = os.path.join(session_dir, 'he_mask.png')
    if not os.path.exists(msi_mask_path) or not os.path.exists(he_mask_path):
        raise FileNotFoundError(
            'Missing msi_mask_original.png or he_mask.png — run rigid registration first.'
        )

    msi_mask = np.array(Image.open(msi_mask_path).convert('L'), dtype=np.float32) / 255.0
    he_mask  = np.array(Image.open(he_mask_path).convert('L'),  dtype=np.float32) / 255.0

    M = np.asarray(affine_matrix, dtype=np.float64)
    if M.shape != (3, 3):
        raise ValueError('affine_matrix must be 3×3.')

    msi_affine = cv2.warpAffine(
        msi_mask.astype(np.float32), M[:2, :].astype(np.float64),
        (he_mask.shape[1], he_mask.shape[0]),
        flags=cv2.INTER_LINEAR, borderValue=0,
    )

    src = np.array([p['src'] for p in pairs], dtype=np.float64)
    dst = np.array([p['dst'] for p in pairs], dtype=np.float64)

    map_x, map_y = build_tps_remap(src, dst, he_mask.shape[:2], reg=TPS_REG)
    msi_tps = cv2.remap(
        msi_affine.astype(np.float32), map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    iou_before = _calc_iou(msi_affine, he_mask)
    iou_after  = _calc_iou(msi_tps,    he_mask)
    disp       = np.sqrt(np.sum((dst - src) ** 2, axis=1))

    overlay      = combine_masks_overlay(he_mask, msi_tps)
    overlay_path = os.path.join(result_dir, 'overlay_tps.png')
    Image.fromarray(overlay, mode='RGB').save(overlay_path)

    trace_path = os.path.join(result_dir, 'coordinate_trace.csv')
    with open(trace_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['idx', 'src_x', 'src_y', 'dst_x', 'dst_y', 'displacement'])
        for i, (s, d, e) in enumerate(zip(src, dst, disp), start=1):
            w.writerow([i, float(s[0]), float(s[1]), float(d[0]), float(d[1]), float(e)])

    return {
        'metrics': {
            'iou_before':        float(iou_before),
            'iou_after':         float(iou_after),
            'mean_displacement': float(disp.mean()) if len(disp) else 0.0,
            'max_displacement':  float(disp.max())  if len(disp) else 0.0,
        },
        'overlay_path':          overlay_path,
        'coordinate_trace_path': trace_path,
        # 控制点对（src 在 affine-warped MSI 空间，dst 在 HE 空间，两者均已对齐到 HE 网格）
        # 供 ROI 模块做 forward TPS warp 用
        'src_points':            src.tolist(),
        'dst_points':            dst.tolist(),
    }
