"""
超分辨模块：加载已训练的 VDSR / HE-VDSR / LCRN-VDSR 权重，直接对输入图像做应用推理。
"""
from __future__ import annotations

import json
import logging
import os

from flask import Blueprint, jsonify, render_template, request, session
from werkzeug.utils import secure_filename

from blueprints.utils import get_h5ad_path, get_sample_dir, img_to_b64, mark_stage_done
from core.super_resolution import (
    MODEL_LABELS,
    MODEL_VARIANTS,
    SuperResolutionConfig,
    apply_super_resolution,
    get_model_status,
    render_mz_channel_image,
)

bp = Blueprint('super_resolution', __name__, url_prefix='/super-resolution')
logger = logging.getLogger(__name__)

ALLOWED_IMG = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}


def _sr_dir() -> str | None:
    sd = get_sample_dir()
    if not sd:
        return None
    d = os.path.join(sd, 'super_resolution')
    os.makedirs(d, exist_ok=True)
    return d


def _rigid_result_path() -> str | None:
    sd = get_sample_dir()
    candidates = []
    if sd:
        candidates.append(os.path.join(sd, 'reg', 'rigid_result.json'))
    sess_p = session.get('rigid_result_json')
    if sess_p:
        candidates.append(sess_p)
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def _load_rigid_he_path() -> str:
    p = _rigid_result_path()
    if not p:
        return ''
    with open(p, 'r', encoding='utf-8') as f:
        data = json.load(f)
    he_path = data.get('he_display_path') or data.get('he_path')
    return he_path if he_path and os.path.exists(he_path) else ''


def _save_upload(field: str, dest_dir: str, required: bool = False) -> str:
    file = request.files.get(field)
    if not file or not file.filename:
        if required:
            raise ValueError(f'请上传 {field}')
        return ''
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_IMG:
        raise ValueError('图像仅支持 .png / .jpg / .tif / .tiff')
    path = os.path.join(dest_dir, secure_filename(file.filename))
    file.save(path)
    return path


@bp.route('/')
def index():
    return render_template(
        'super_resolution.html',
        active='super_resolution',
        variants=MODEL_VARIANTS,
        labels=MODEL_LABELS,
    )


@bp.route('/api/status')
def status():
    he_path = _load_rigid_he_path()
    h5ad_path = get_h5ad_path()
    mz_info = _summarize_h5ad_mz(h5ad_path)
    sr_dir = _sr_dir()
    summary_path = os.path.join(sr_dir, 'super_resolution_result.json') if sr_dir else ''
    result = None
    if summary_path and os.path.exists(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
    return jsonify({
        'status': 'ok',
        'has_rigid': bool(he_path),
        'rigid_he_name': os.path.basename(he_path) if he_path else '',
        'has_msi_data': bool(h5ad_path),
        'h5ad_name': os.path.basename(h5ad_path) if h5ad_path else '',
        'mz_info': mz_info,
        'models': get_model_status(),
        'has_result': result is not None,
        'result': _summarize_result(result) if result else None,
    })


@bp.route('/api/run', methods=['POST'])
def run():
    sr_dir = _sr_dir()
    if not sr_dir:
        return jsonify({'status': 'error', 'message': '没有活跃样本，请到「样本管理」'}), 400

    input_dir = os.path.join(sr_dir, 'inputs')
    os.makedirs(input_dir, exist_ok=True)

    try:
        variant = request.form.get('model_variant', 'lcrn_guided')
        input_source = request.form.get('input_source', 'upload')
        generated_input = None
        if input_source == 'mz_channel':
            h5ad_path = get_h5ad_path()
            if not h5ad_path:
                raise ValueError('当前样本未找到 h5ad 数据，请先在「数据导入」模块上传并归一化')
            target_raw = request.form.get('target_mz', '').strip()
            if not target_raw:
                raise ValueError('请输入目标 m/z')
            target_mz = float(target_raw)
            tolerance_raw = request.form.get('mz_tolerance', '').strip()
            tolerance = float(tolerance_raw) if tolerance_raw else None
            generated_input = render_mz_channel_image(
                h5ad_path=h5ad_path,
                target_mz=target_mz,
                tolerance=tolerance,
                output_dir=input_dir,
            )
            lr_path = generated_input['image_path']
        else:
            lr_path = _save_upload('lr_msi_file', input_dir, required=True)
        uploaded_he = _save_upload('he_file', input_dir, required=False)
        he_path = uploaded_he or _load_rigid_he_path()
        cfg = SuperResolutionConfig(
            lr_msi_path=lr_path,
            he_path=he_path,
            output_dir=sr_dir,
            model_variant=variant,
            upscale_factor=max(1, int(request.form.get('upscale_factor', 1))),
        )
        result = apply_super_resolution(cfg)
        if generated_input:
            result['generated_input'] = generated_input
            result['input_source'] = input_source
            summary_path = result.get('summary_path')
            if summary_path:
                with open(summary_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.exception('super-resolution inference error')
        return jsonify({'status': 'error', 'message': str(e)}), 500

    mark_stage_done('super_resolution')
    return jsonify({'status': 'success', 'result': _summarize_result(result)})


def _summarize_h5ad_mz(h5ad_path: str | None) -> dict | None:
    if not h5ad_path or not os.path.exists(h5ad_path):
        return None
    try:
        import anndata as ad
        import numpy as np

        adata = ad.read_h5ad(h5ad_path, backed='r')
        if 'm/z' in adata.var.columns:
            mz_values = adata.var['m/z'].astype(str).astype(float).to_numpy()
        else:
            mz_values = np.asarray([float(str(v).strip()) for v in adata.var_names], dtype=float)
        finite = mz_values[np.isfinite(mz_values)]
        if finite.size == 0:
            return None
        return {
            'count': int(finite.size),
            'min': float(finite.min()),
            'max': float(finite.max()),
            'examples': [float(v) for v in finite[:5]],
        }
    except Exception:
        logger.exception('summarize h5ad mz error')
        return None


def _summarize_result(result: dict | None) -> dict | None:
    if not result:
        return None
    preview = result.get('preview_path') or result.get('output_path') or ''
    lr_preview = result.get('lr_preview_path') or ''
    return {
        'finished_at': result.get('finished_at'),
        'variant': result.get('variant'),
        'variant_label': result.get('variant_label'),
        'device': result.get('device'),
        'model_path': result.get('model_path'),
        'input_path': result.get('input_path'),
        'he_path': result.get('he_path'),
        'input_size': result.get('input_size'),
        'output_size': result.get('output_size'),
        'lr_preview_size': result.get('lr_preview_size'),
        'upscale_factor': result.get('upscale_factor'),
        'he_resized': result.get('he_resized'),
        'output_path': result.get('output_path'),
        'preview_path': preview,
        'preview_img': img_to_b64(preview) if preview else None,
        'lr_preview_path': lr_preview,
        'lr_preview_img': img_to_b64(lr_preview) if lr_preview else None,
        'summary_path': result.get('summary_path'),
        'input_source': result.get('input_source'),
        'generated_input': result.get('generated_input'),
    }
