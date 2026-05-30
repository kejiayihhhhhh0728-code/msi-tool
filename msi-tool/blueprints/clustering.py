"""
模块3：降维与空间聚类
====================
输入：模块1归一化后的 h5ad（session['h5ad_norm_tic'] 或 h5ad_norm_rms，fallback raw）
输出：7 种方法并列的聚类标签 + 空间图 + UMAP 可视化图 + 指标表 + CSV
依赖：模块1（数据导入与归一化）
"""
import logging
import os

import pandas as pd
from flask import (Blueprint, render_template, request, jsonify,
                   session, send_file)

logger = logging.getLogger(__name__)

from core.clustering_algo import run_clustering_pipeline
from config import UPLOAD_FOLDER
from blueprints.utils import get_h5ad_path

bp = Blueprint('clustering', __name__, url_prefix='/clustering')


# ─── 页面 ─────────────────────────────────────────────────────────────

@bp.route('/')
def index():
    return render_template('clustering.html', active='clustering')


# ─── API：执行聚类 ────────────────────────────────────────────────────

@bp.route('/api/run', methods=['POST'])
def run_clustering():
    """
    执行聚类流程，返回图片 base64 + 指标。

    JSON body（所有字段可选，有默认值）:
      n_clusters                int    聚类数，默认 5
      dr_method                 str    降维路线 'pca' 或 'umap_dr'，默认 'pca'
      pca_n_components          int    PCA 固定维度，默认 30
      pca_variance_threshold    float  PCA 累积方差阈值（设置后覆盖 pca_n_components）
      umap_dr_n_components      int    UMAP-DR 目标维度，默认 10
      umap_dr_n_neighbors       int    UMAP-DR 邻居数，默认 30
      umap_dr_metric            str    UMAP-DR 距离度量，默认 'cosine'
      spatial_radius            float  空间平滑半径，默认 0.0（关闭）
      spatial_alpha             float  空间平滑权重，默认 0.0（关闭）
      median_kernel             int    中值滤波核（0=关闭），默认 0
      do_umap_vis               bool   是否生成可视化 UMAP，默认 true
      gmm_auto_bic              bool   GMM 是否用 BIC 自动选 k，默认 false
      gmm_search_range          int    BIC 扫描范围 ±，默认 5
      hdbscan_min_cluster_size  int    HDBSCAN 最小簇大小，默认 100
      hdbscan_min_samples       int    HDBSCAN 最小样本数，默认 30
    """
    h5ad_path = get_h5ad_path()
    if not h5ad_path:
        return jsonify({'status': 'error',
                        'message': '请先在「数据导入」模块上传并归一化数据'}), 400

    body = request.get_json(silent=True) or {}

    # pca_variance_threshold：前端传 0 表示不设置
    pca_vt_raw = body.get('pca_variance_threshold', 0)
    pca_vt = float(pca_vt_raw) if pca_vt_raw and float(pca_vt_raw) > 0 else None

    params = {
        'n_clusters':               int(body.get('n_clusters', 5)),
        'dr_method':                str(body.get('dr_method', 'pca')),
        'pca_n_components':         int(body.get('pca_n_components', 30)),
        'pca_variance_threshold':   pca_vt,
        'umap_dr_n_components':     int(body.get('umap_dr_n_components', 10)),
        'umap_dr_n_neighbors':      int(body.get('umap_dr_n_neighbors', 30)),
        'umap_dr_metric':           str(body.get('umap_dr_metric', 'cosine')),
        'spatial_radius':           float(body.get('spatial_radius', 0.0)),
        'spatial_alpha':            float(body.get('spatial_alpha', 0.0)),
        'median_kernel':            int(body.get('median_kernel', 0)),
        'do_umap_vis':              bool(body.get('do_umap_vis', True)),
        'gmm_auto_bic':             bool(body.get('gmm_auto_bic', False)),
        'gmm_search_range':         int(body.get('gmm_search_range', 5)),
        'hdbscan_min_cluster_size': int(body.get('hdbscan_min_cluster_size', 100)),
        'hdbscan_min_samples':      int(body.get('hdbscan_min_samples', 30)),
    }

    try:
        out = run_clustering_pipeline(h5ad_path, **params)

        # 保存 CSV 供下载
        sid = session.get('sid')
        csv_available = False
        if sid:
            sid_dir = os.path.join(UPLOAD_FOLDER, sid)
            csv_path = os.path.join(sid_dir, 'clustering_results.csv')
            out['results_df'].to_csv(csv_path, index=False)
            session['clustering_csv'] = csv_path
            csv_available = True

        return jsonify({
            'status':        'success',
            'spatial_img':   out['spatial_img'],
            'umap_img':      out['umap_img'],
            'metrics':       out['metrics'],
            'info':          out['info'],
            'csv_available': csv_available,
        })

    except Exception as e:
        logger.exception('clustering error')
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ─── API：下载 CSV ────────────────────────────────────────────────────

@bp.route('/api/download_csv')
def download_csv():
    csv_path = session.get('clustering_csv')
    if not csv_path or not os.path.exists(csv_path):
        return jsonify({'status': 'error', 'message': '请先执行聚类'}), 404
    return send_file(csv_path, as_attachment=True,
                     download_name='clustering_results.csv',
                     mimetype='text/csv')
