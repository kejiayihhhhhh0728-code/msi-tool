import os
import math
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from core.reg_step1 import analyze_contour
from core.reg_step2 import (
    register_iou_mi, run_icp, pixel_iou, warp_mask,
    create_tic_image, he_to_grayscale, _compute_mi,
)

MI_WEIGHT    = 0.3
MI_N_BINS    = 64
ICP_MAX_ITER = 50
ICP_TOLERANCE = 0.01


def _create_msi_mask_from_txt(file_path: str) -> Tuple[np.ndarray, pd.DataFrame]:
    data = pd.read_table(file_path, sep='\t')
    h = int(data['y'].max() + 1)
    w = int(data['x'].max() + 1)
    mask = np.zeros((h, w), dtype=np.float32)
    mask[data['y'].values.astype(int), data['x'].values.astype(int)] = 1.0
    return mask, data


def _load_he_image(image_path: str) -> Tuple[np.ndarray, np.ndarray]:
    rgba  = np.array(Image.open(image_path).convert('RGBA'))
    alpha = rgba[:, :, 3].astype(np.float32) / 255.0
    rgb   = rgba[:, :, :3].astype(np.float32)
    for c in range(3):
        rgb[:, :, c] = rgb[:, :, c] * alpha + 255.0 * (1.0 - alpha)
    return rgb / 255.0, (rgba[:, :, 3] > 0).astype(np.float32)


def combine_masks_overlay(hemask: np.ndarray, msi_warped: np.ndarray) -> np.ndarray:
    """Combine HE and MSI masks into a colour overlay (blue=HE only, red=MSI only, white=overlap)."""
    out = np.zeros(hemask.shape[:2] + (3,), dtype=np.uint8)
    he  = hemask > 0.5
    ms  = msi_warped > 0.5
    out[he & ~ms] = [46, 99, 161]
    out[~he & ms] = [135, 56, 62]
    out[he & ms]  = [255, 255, 255]
    return out


def _save_overlay(he_mask, msi_mask_bin, M, dst_shape, save_path) -> Tuple[str, Dict]:
    msi_warped     = warp_mask(msi_mask_bin, M, dst_shape)
    msi_warped_bin = (msi_warped > 0.5).astype(np.float32)
    iou, overlap, union = pixel_iou(msi_mask_bin, he_mask, M)

    overlay = combine_masks_overlay(he_mask, msi_warped_bin)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Image.fromarray(overlay, mode='RGB').save(save_path)

    he_bin = he_mask > 0.5
    ms_bin = msi_warped_bin > 0.5
    tp = float(np.sum(he_bin & ms_bin))
    fp = float(np.sum(~he_bin & ms_bin))
    fn = float(np.sum(he_bin & ~ms_bin))
    tn = float(np.sum(~he_bin & ~ms_bin))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    dsc       = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0.0
    coverage  = tp / float(np.sum(he_bin)) if np.sum(he_bin) > 0 else 0.0
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return save_path, {
        'iou': float(iou), 'coverage': float(coverage), 'dsc': float(dsc),
        'precision': float(precision), 'recall': float(recall), 'fpr': float(fpr),
        'overlap': float(overlap), 'union': float(union),
    }


def run_multimodal_registration(msi_path: str, he_path: str, output_dir: str) -> Dict:
    """
    执行完整刚性配准流程。
    Returns: affine_matrix, metrics, overlay_path, tic_warped_path, he_display_path
    """
    os.makedirs(output_dir, exist_ok=True)

    msi_mask_bin, msi_df = _create_msi_mask_from_txt(msi_path)
    he_rgb_01, he_mask_bin = _load_he_image(he_path)

    msi_mask_png    = os.path.join(output_dir, 'msi_mask_original.png')
    he_mask_png     = os.path.join(output_dir, 'he_mask.png')
    he_display_path = os.path.join(output_dir, 'he_display.png')

    Image.fromarray((msi_mask_bin * 255).astype(np.uint8), mode='L').save(msi_mask_png)
    Image.fromarray((he_mask_bin  * 255).astype(np.uint8), mode='L').save(he_mask_png)
    Image.fromarray((np.clip(he_rgb_01, 0, 1) * 255).astype(np.uint8), mode='RGB').save(he_display_path)

    analysis_dir = os.path.join(output_dir, 'contour_analysis')
    msi_geo = analyze_contour(msi_mask_png, analysis_dir, visualize=True)
    he_geo  = analyze_contour(he_path,      analysis_dir, visualize=True)
    if msi_geo is None or he_geo is None:
        raise RuntimeError('Contour analysis failed for MSI or HE image.')

    tic_image = create_tic_image(msi_df, msi_mask_bin.shape)
    he_gray   = he_to_grayscale(he_path)

    M_best, reg_results = register_iou_mi(
        msi_geo, he_geo, msi_mask_bin, he_mask_bin,
        tic_image, he_gray,
        mi_weight=MI_WEIGHT, n_bins=MI_N_BINS,
    )

    tic_norm = tic_image.astype(np.float64)
    if tic_norm.max() > 0: tic_norm /= tic_norm.max()
    he_norm  = he_gray.astype(np.float64)
    if he_norm.max() > 0:  he_norm  /= he_norm.max()

    iou_no_icp = pixel_iou(msi_mask_bin, he_mask_bin, M_best)[0]
    mi_no_icp  = _compute_mi(tic_norm, he_norm, msi_mask_bin, he_mask_bin, M_best, n_bins=MI_N_BINS)
    best_score = iou_no_icp * (1 - MI_WEIGHT) + mi_no_icp * MI_WEIGHT
    best_M     = M_best

    for M_cand, _, _ in reg_results.get('candidates', []):
        M_icp, _ = run_icp(msi_geo, he_geo, M_cand, msi_mask_bin, he_mask_bin, ICP_MAX_ITER, ICP_TOLERANCE)
        iou_post = pixel_iou(msi_mask_bin, he_mask_bin, M_icp)[0]
        mi_post  = _compute_mi(tic_norm, he_norm, msi_mask_bin, he_mask_bin, M_icp, n_bins=MI_N_BINS)
        score    = iou_post * (1 - MI_WEIGHT) + mi_post * MI_WEIGHT
        if score > best_score:
            best_score = score
            best_M     = M_icp

    overlay_path, metrics = _save_overlay(
        he_mask_bin, msi_mask_bin, best_M, he_mask_bin.shape,
        os.path.join(output_dir, 'overlay_registration.png'),
    )

    tic_warped = cv2.warpAffine(
        tic_image.astype(np.float32), best_M[:2, :].astype(np.float64),
        (he_mask_bin.shape[1], he_mask_bin.shape[0]),
        flags=cv2.INTER_LINEAR, borderValue=0,
    )
    tic_vis = tic_warped / tic_warped.max() if tic_warped.max() > 0 else tic_warped
    tic_warped_path = os.path.join(output_dir, 'tic_warped.png')
    Image.fromarray((tic_vis * 255).astype(np.uint8), mode='L').save(tic_warped_path)

    np.save(os.path.join(output_dir, 'best_affine_matrix.npy'), best_M)

    return {
        'affine_matrix':   best_M.tolist(),
        'metrics':         {k: metrics[k] for k in ('iou', 'coverage', 'dsc', 'precision', 'recall', 'fpr')},
        'overlay_path':    overlay_path,
        'tic_warped_path': tic_warped_path,
        'he_display_path': he_display_path,
    }
