"""模块10：差异代谢物空间热图。"""
from __future__ import annotations

from flask import Blueprint, render_template, request, jsonify

from blueprints.utils import ensure_active_batch, get_batch_dir
from core.cohort import assemble_cohort
from core.visualization import load_diff_metabolites, collect_metabolite_whole_slice_pixels, render_batch_heatmap

bp = Blueprint('spatial_heatmap', __name__, url_prefix='/heatmap')


@bp.route('/')
def index():
    return render_template('heatmap.html', active='spatial_heatmap')


@bp.route('/api/status')
def status():
    ensure_active_batch()
    batch_dir = get_batch_dir()
    tissues = []
    if batch_dir:
        for name, data in assemble_cohort(batch_dir).items():
            mets = load_diff_metabolites(batch_dir, name, max_items=100)
            tissues.append({
                'name': name,
                'n_samples': data['n_samples'],
                'metabolites': [
                    {
                        'id': str(r.get('Metabolite')),
                        'log2fc': r.get('mean_log2FC'),
                        'fdr': r.get('FDR'),
                        'change': r.get('final_change', r.get('change', '')),
                    }
                    for _, r in mets.iterrows()
                ],
            })
    return jsonify({'status': 'ok', 'tissues': tissues})


@bp.route('/api/render', methods=['POST'])
def render_heatmap():
    ensure_active_batch()
    batch_dir = get_batch_dir()
    if not batch_dir:
        return jsonify({'status': 'error', 'message': '未找到当前批次'}), 400
    body = request.get_json(silent=True) or {}
    tissue = (body.get('tissue') or '').strip()
    metabolite = (body.get('metabolite') or '').strip()
    cmap = body.get('colormap') or 'magma'
    if not tissue or not metabolite:
        return jsonify({'status': 'error', 'message': '请选择 tissue 和代谢物'}), 400
    items = collect_metabolite_whole_slice_pixels(batch_dir, tissue, metabolite)
    if not items:
        return jsonify({'status': 'error', 'message': '没有找到该代谢物的整张切片像素数据'}), 404
    img = render_batch_heatmap(items, metabolite, colormap=cmap)
    if not img:
        return jsonify({'status': 'error', 'message': '热图渲染失败'}), 500
    return jsonify({
        'status': 'ok',
        'tissue': tissue,
        'metabolite': metabolite,
        'n_panels': len(items),
        'heatmap_img': img,
    })
