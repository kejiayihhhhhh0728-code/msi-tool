"""
模块1：数据导入与预处理
输入：MSI 原始数据（分号分隔 csv，行为 m/z，列为 spot）+ spot 坐标文件
输出：归一化后的 AnnData（存为 .h5ad 到当前样本目录）
依赖：批次/样本系统（blueprints.batch + blueprints.utils）

所有上传文件都先经 SHA-256 缓存（library/upload_cache/）跨批次去重，
再 hardlink/copy 到当前样本目录。h5ad 产物落在样本目录，模块完成后
通过 mark_stage_done 在 batch_meta.json 中登记进度。
"""
import logging
import os

import anndata as ad
from flask import Blueprint, render_template, request, jsonify, session
from werkzeug.utils import secure_filename

logger = logging.getLogger(__name__)

from core.preprocessing import ms_to_anndata, normalize_adata, plot_tic_histogram, get_adata_stats
from blueprints.utils import (
    get_sample_dir, load_sample_meta, save_sample_meta, mark_stage_done,
    cache_upload, materialize_into_sample,
)

bp = Blueprint('data_import', __name__, url_prefix='/data-import')

ALLOWED = {'.csv', '.txt', '.tsv'}


def _allowed(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in ALLOWED


# ─── 页面 ─────────────────────────────────────────────────────

@bp.route('/')
def index():
    return render_template('data_import.html', active='data_import')


# ─── API：上传并解析 ──────────────────────────────────────────

@bp.route('/api/upload', methods=['POST'])
def upload():
    """
    上传 MSI 数据 + spot 坐标，落到当前样本目录，解析为 raw.h5ad。
    """
    msi_f = request.files.get('msi_file')
    spot_f = request.files.get('spot_file')
    resolution = float(request.form.get('resolution', 20))

    if not msi_f or not msi_f.filename:
        return jsonify({'status': 'error', 'message': '请上传 MSI 数据文件'}), 400
    if not _allowed(msi_f.filename):
        return jsonify({'status': 'error', 'message': 'MSI 文件只支持 .csv / .txt 格式'}), 400
    if spot_f and spot_f.filename and not _allowed(spot_f.filename):
        return jsonify({'status': 'error', 'message': 'Spot 文件只支持 .csv / .txt 格式'}), 400

    sample_dir = get_sample_dir()
    if not sample_dir:
        return jsonify({'status': 'error',
                        'message': '请先在「样本管理」页面添加样本'}), 400

    # 1) MSI 文件走 SHA-256 缓存，再 hardlink/copy 到样本目录
    try:
        msi_ext = os.path.splitext(msi_f.filename)[1].lower()
        sha16, cache_path = cache_upload(msi_f, ext_hint=msi_ext)
        msi_path = materialize_into_sample(cache_path, sample_dir, f'msi_raw{msi_ext}')
    except Exception as e:
        logger.exception('msi cache/materialize error')
        return jsonify({'status': 'error', 'message': f'保存 MSI 失败: {e}'}), 500

    # 2) Spot 文件可选：直接保存到样本目录（一般体积小，无需缓存）
    spot_path = None
    if spot_f and spot_f.filename:
        spot_path = os.path.join(sample_dir, secure_filename(spot_f.filename))
        spot_f.save(spot_path)

    # 3) 解析为 AnnData，落盘 raw.h5ad
    h5ad_path = os.path.join(sample_dir, 'raw.h5ad')
    try:
        adata = ms_to_anndata(
            ms_file=msi_path,
            spot_file=spot_path,
            resolution=resolution,
            save_path=h5ad_path,
        )
    except Exception as e:
        logger.exception('parse error')
        return jsonify({'status': 'error', 'message': f'解析失败: {e}'}), 500

    # 4) 更新样本元数据 + 阶段标记
    meta = load_sample_meta() or {}
    meta['msi_filename'] = msi_f.filename
    meta['msi_sha16']    = sha16
    meta['resolution']   = resolution
    save_sample_meta(meta)
    mark_stage_done('import')

    stats = get_adata_stats(adata)
    return jsonify({
        'status': 'success',
        'message': f'数据解析完成：{stats["n_pixels"]} 个像素 × {stats["n_mz"]} 个 m/z',
        'sample_id': meta.get('id', ''),
        'stats': stats,
    })


# ─── API：执行归一化 ──────────────────────────────────────────

@bp.route('/api/normalize', methods=['POST'])
def normalize():
    """
    对当前样本的 raw.h5ad 执行 TIC 或 RMS 归一化，
    输出到 norm_<method>.h5ad。
    """
    data = request.get_json(silent=True) or {}
    method = data.get('method', 'TIC').upper()

    if method not in ('TIC', 'RMS'):
        return jsonify({'status': 'error', 'message': 'method 必须为 TIC 或 RMS'}), 400

    sample_dir = get_sample_dir()
    if not sample_dir:
        return jsonify({'status': 'error', 'message': '没有活跃样本'}), 400

    h5ad_raw  = os.path.join(sample_dir, 'raw.h5ad')
    h5ad_norm = os.path.join(sample_dir, f'norm_{method.lower()}.h5ad')

    if not os.path.exists(h5ad_raw):
        return jsonify({'status': 'error',
                        'message': '原始数据不存在，请先上传'}), 400

    try:
        adata_raw  = ad.read_h5ad(h5ad_raw)
        adata_norm = normalize_adata(adata_raw, method=method, inplace=False)
        adata_norm.write(h5ad_norm)
    except Exception as e:
        logger.exception('normalize error')
        return jsonify({'status': 'error', 'message': str(e)}), 500

    # 更新样本元数据（记录归一化方式，供其他模块默认使用）
    meta = load_sample_meta() or {}
    meta['norm_method'] = method
    save_sample_meta(meta)
    mark_stage_done('norm')

    hist_b64 = plot_tic_histogram(adata_raw, adata_norm, method=method)
    stats_norm = get_adata_stats(adata_norm)

    return jsonify({
        'status': 'success',
        'message': f'{method} 归一化完成',
        'method': method,
        'hist_img': hist_b64,
        'stats': stats_norm,
    })
