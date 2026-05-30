"""
模块4：空间积累模式分析（NMF）
================================
输入：模块1归一化后的 h5ad（session 中）
输出：NMF 空间模式图 + Spearman 相关 top 代谢物列表 + CSV

依赖：模块1（数据导入与归一化）
"""
import logging
import os

import pandas as pd
from flask import (Blueprint, render_template, request, jsonify,
                   session, send_file)

logger = logging.getLogger(__name__)

from core.nmf_analysis import run_nmf_pipeline
from config import UPLOAD_FOLDER
from blueprints.utils import get_h5ad_path

bp = Blueprint('nmf_patterns', __name__, url_prefix='/nmf')


# ─── 页面 ─────────────────────────────────────────────────────────────────────

@bp.route('/')
def index():
    return render_template('nmf.html', active='nmf_patterns')


# ─── API：执行 NMF ─────────────────────────────────────────────────────────────

@bp.route('/api/run', methods=['POST'])
def run_nmf():
    """
    执行空间 NMF 积累模式分析。

    JSON body（所有字段可选，有默认值）:
      n_components  int    NMF 分量数（模式数），默认 5
      cutoff        float  Moran's I FDR 截断阈值，默认 0.5
      n_neighbors   int    Moran's I k-NN 邻居数，默认 10
      top_n         int    每模式展示的代表代谢物数，默认 10
      moran_filter  bool   是否进行 Moran's I 筛选，默认 true
      max_cutoff    float  空间图颜色分位数截断，默认 0.9

    返回 JSON:
      status           'success' | 'error'
      pattern_grid_img str    合并网格图（base64 PNG）
      pattern_imgs     list   各模式单独图（list of base64 PNG）
      top_metabolites  list   每模式 top 代谢物（按 Spearman ρ 排序）
      top_met_imgs     list   每模式 top 代谢物空间分布图（base64 PNG）
      info             dict   统计信息
      csv_available    bool
    """
    h5ad_path = get_h5ad_path()
    if not h5ad_path:
        return jsonify({
            'status':  'error',
            'message': '请先在「数据导入」模块上传并归一化数据',
        }), 400

    body = request.get_json(silent=True) or {}

    n_components = int(body.get('n_components', 5))
    cutoff       = float(body.get('cutoff', 0.5))
    n_neighbors  = int(body.get('n_neighbors', 10))
    top_n        = int(body.get('top_n', 10))
    moran_filter = bool(body.get('moran_filter', True))
    max_cutoff   = float(body.get('max_cutoff', 0.9))

    # 参数范围保护
    n_components = max(2, min(n_components, 30))
    cutoff       = max(0.001, min(cutoff, 1.0))
    n_neighbors  = max(3, min(n_neighbors, 50))
    top_n        = max(3, min(top_n, 50))
    max_cutoff   = max(0.5, min(max_cutoff, 1.0))

    try:
        result = run_nmf_pipeline(
            h5ad_path=h5ad_path,
            n_components=n_components,
            cutoff=cutoff,
            n_neighbors=n_neighbors,
            top_n=top_n,
            moran_filter=moran_filter,
            max_cutoff=max_cutoff,
        )

        # 保存 Spearman CSV 供下载
        sid = session.get('sid')
        csv_available = False
        if sid and result['spearman_table']:
            sid_dir  = os.path.join(UPLOAD_FOLDER, sid)
            os.makedirs(sid_dir, exist_ok=True)
            csv_path = os.path.join(sid_dir, 'nmf_spearman.csv')
            pd.DataFrame(result['spearman_table']).to_csv(csv_path, index=False)
            session['nmf_csv'] = csv_path
            csv_available = True

        return jsonify({
            'status':           'success',
            'pattern_grid_img': result['pattern_grid_img'],
            'pattern_imgs':     result['pattern_imgs'],
            'top_metabolites':  result['top_metabolites'],
            'top_met_imgs':     result['top_met_imgs'],
            'info':             result['info'],
            'csv_available':    csv_available,
        })

    except Exception as e:
        logger.exception('nmf error')
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ─── API：下载 CSV ─────────────────────────────────────────────────────────────

@bp.route('/api/download_csv')
def download_csv():
    """下载 Spearman 相关表 CSV（每模式 top 代谢物 × 4 列）"""
    csv_path = session.get('nmf_csv')
    if not csv_path or not os.path.exists(csv_path):
        return jsonify({'status': 'error', 'message': '请先执行 NMF 分析'}), 404
    return send_file(
        csv_path,
        as_attachment=True,
        download_name='nmf_spearman.csv',
        mimetype='text/csv',
    )
