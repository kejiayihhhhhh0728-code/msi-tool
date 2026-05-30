"""模块11：通路富集分析。"""
from __future__ import annotations

import os

import numpy as np
from flask import Blueprint, render_template, request, jsonify, send_file

from blueprints.utils import ensure_active_batch, get_batch_dir
from core.cohort import assemble_cohort
from core.pathway import run_pathway_pipeline, run_spatial_pathway_pipeline, safe_name

bp = Blueprint('pathway', __name__, url_prefix='/pathway')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_GMT = os.path.join(BASE_DIR, 'data', 'pathways_metabolite_selected85.gmt')


@bp.route('/')
def index():
    return render_template('pathway.html', active='pathway')


@bp.route('/api/status')
def status():
    ensure_active_batch()
    batch_dir = get_batch_dir()
    tissues = []
    if batch_dir:
        for name, data in assemble_cohort(batch_dir).items():
            diff_csv = os.path.join(batch_dir, 'diff', safe_name(name), 'differential_metabolites.csv')
            tissues.append({
                'name': name,
                'n_samples': data['n_samples'],
                'has_diff': os.path.exists(diff_csv),
            })
    return jsonify({
        'status': 'ok',
        'tissues': tissues,
        'gmt_available': os.path.exists(DEFAULT_GMT),
        'gmt_path': DEFAULT_GMT,
    })


@bp.route('/api/run', methods=['POST'])
def run():
    ensure_active_batch()
    batch_dir = get_batch_dir()
    if not batch_dir:
        return jsonify({'status': 'error', 'message': '未找到当前批次'}), 400
    if not os.path.exists(DEFAULT_GMT):
        return jsonify({'status': 'error', 'message': '缺少 GMT 文件'}), 500

    body = request.get_json(silent=True) or {}
    tissue = (body.get('tissue') or '').strip()
    if not tissue:
        cohorts = assemble_cohort(batch_dir)
        tissue = sorted(cohorts.keys())[0] if cohorts else ''
    if not tissue:
        return jsonify({'status': 'error', 'message': '当前批次没有可分析 tissue'}), 400

    result = run_pathway_pipeline(
        batch_dir, tissue, DEFAULT_GMT,
        fc_cutoff=float(body.get('fc_cutoff') or np.log2(1.2)),
        fdr_cutoff=float(body.get('fdr_cutoff') or 0.05),
        use_vip=bool(body.get('use_vip', False)),
        vip_cutoff=float(body.get('vip_cutoff') or 1.0),
        pathway_sig_cutoff=float(body.get('pathway_sig_cutoff') or 0.05),
    )
    if result.get('error'):
        return jsonify({'status': 'error', 'message': result['error']}), 400

    out_dir = os.path.join(batch_dir, 'pathway', safe_name(tissue))
    os.makedirs(out_dir, exist_ok=True)
    result['diff_df'].to_csv(os.path.join(out_dir, 'diff_with_kegg.csv'), index=False, encoding='utf-8-sig')

    response_outputs = {}
    for change, payload in result['outputs'].items():
        df = payload['result_df']
        if not df.empty:
            df.to_csv(os.path.join(out_dir, f'{change}_enrichment.csv'), index=False, encoding='utf-8-sig')
        clean_df = df.replace([np.inf, -np.inf], np.nan).where(df.notna(), None) if not df.empty else df
        response_outputs[change] = {
            'n_genes': len(payload['genes']),
            'dotplot_img': payload['dotplot_img'],
            'table': clean_df.head(50).to_dict(orient='records') if not clean_df.empty else [],
        }

    return jsonify({
        'status': 'ok',
        'tissue': tissue,
        'background_size': result['background_size'],
        'mapped_metabolites': result['mapped_metabolites'],
        'gmt_pathways': result['gmt_pathways'],
        'sig_pathways': result['sig_pathways'],
        'outputs': response_outputs,
        'export_ready': True,
    })


@bp.route('/api/spatial', methods=['POST'])
def spatial_pathway():
    """Mode B：将 Mode A 的显著通路映射回每个样本的全像素空间，生成活性热图。"""
    ensure_active_batch()
    batch_dir = get_batch_dir()
    if not batch_dir:
        return jsonify({'status': 'error', 'message': '未找到当前批次'}), 400
    if not os.path.exists(DEFAULT_GMT):
        return jsonify({'status': 'error', 'message': '缺少 GMT 文件'}), 500

    body = request.get_json(silent=True) or {}
    pathways = body.get('pathways') or []
    pathways = [p for p in pathways if isinstance(p, str) and p.strip()]
    if not pathways:
        return jsonify({'status': 'error', 'message': '请至少选择一条通路'}), 400
    sample_ids = body.get('sample_ids') or None
    if sample_ids:
        sample_ids = [s for s in sample_ids if isinstance(s, str)]

    tissue = (body.get('tissue') or '').strip()
    save_dir = os.path.join(batch_dir, 'pathway', safe_name(tissue) if tissue else 'spatial', 'spatial') if tissue else None

    result = run_spatial_pathway_pipeline(
        batch_dir,
        pathway_names=pathways,
        gmt_path=DEFAULT_GMT,
        sample_ids=sample_ids,
        mz_tolerance=float(body.get('mz_tolerance') or 0.005),
        save_dir=save_dir,
    )
    if result.get('error'):
        return jsonify({'status': 'error', 'message': result['error']}), 400

    overlap_df = result.get('pathway_overlap')
    overlap_records = []
    if overlap_df is not None and not overlap_df.empty:
        overlap_records = overlap_df.to_dict(orient='records')

    return jsonify({
        'status': 'ok',
        'n_samples': result.get('n_samples', 0),
        'samples': result.get('samples', []),
        'pathway_overlap': overlap_records,
        'errors': result.get('errors', []),
    })


@bp.route('/api/export', methods=['POST'])
def export():
    ensure_active_batch()
    batch_dir = get_batch_dir()
    body = request.get_json(silent=True) or {}
    tissue = (body.get('tissue') or '').strip()
    if not batch_dir or not tissue:
        return jsonify({'status': 'error', 'message': '缺少 tissue'}), 400
    out_dir = os.path.join(batch_dir, 'pathway', safe_name(tissue))
    xlsx = os.path.join(out_dir, 'pathway_enrichment.xlsx')
    if not os.path.exists(out_dir):
        return jsonify({'status': 'error', 'message': '请先运行通路富集'}), 404

    import pandas as pd
    with pd.ExcelWriter(xlsx) as writer:
        for name in ['diff_with_kegg', 'Up_enrichment', 'Down_enrichment']:
            csv = os.path.join(out_dir, f'{name}.csv')
            if os.path.exists(csv):
                pd.read_csv(csv).to_excel(writer, sheet_name=name[:31], index=False)
    return send_file(xlsx, as_attachment=True, download_name=f'pathway_{safe_name(tissue)}.xlsx')

