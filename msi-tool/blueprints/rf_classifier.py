"""模块9：RF 组合分类器与 Biomarker 筛选。"""
from __future__ import annotations

import os

import numpy as np
from flask import Blueprint, render_template, request, jsonify, send_file

from blueprints.utils import ensure_active_batch, get_batch_dir
from core.cohort import assemble_cohort
from core.classifier import run_classifier_pipeline, _safe_name

bp = Blueprint('rf_classifier', __name__, url_prefix='/classifier')


@bp.route('/')
def index():
    return render_template('classifier.html', active='rf_classifier')


@bp.route('/api/status')
def status():
    ensure_active_batch()
    batch_dir = get_batch_dir()
    tissues = []
    if batch_dir:
        for name, data in assemble_cohort(batch_dir).items():
            diff_csv = os.path.join(batch_dir, 'diff', _safe_name(name), 'differential_metabolites.csv')
            tissues.append({
                'name': name,
                'n_samples': data['n_samples'],
                'n_metabolites': int(data['cancer_mat'].shape[0]),
                'has_diff': os.path.exists(diff_csv),
            })
    return jsonify({'status': 'ok', 'tissues': tissues})


@bp.route('/api/run', methods=['POST'])
def run():
    ensure_active_batch()
    batch_dir = get_batch_dir()
    if not batch_dir:
        return jsonify({'status': 'error', 'message': '未找到当前批次'}), 400
    body = request.get_json(silent=True) or {}
    tissue = (body.get('tissue') or '').strip()
    if not tissue:
        cohorts = assemble_cohort(batch_dir)
        tissue = sorted(cohorts.keys())[0] if cohorts else ''
    if not tissue:
        return jsonify({'status': 'error', 'message': '当前批次没有可分析 tissue'}), 400

    top_n_list = [int(x) for x in body.get('top_n_list', [3, 5, 10, 15]) if int(x) > 0]
    result = run_classifier_pipeline(
        batch_dir, tissue,
        top_n_list=top_n_list,
        n_estimators=int(body.get('n_estimators') or 200),
        n_permutations=int(body.get('n_permutations') or 20),
        max_features=int(body.get('max_features') or 30),
        max_pixels_per_class=int(body.get('max_pixels_per_class') or 20000),
    )
    if result.get('error'):
        return jsonify({'status': 'error', 'message': result['error']}), 400

    out_dir = os.path.join(batch_dir, 'classifier', _safe_name(tissue))
    os.makedirs(out_dir, exist_ok=True)
    decision_path = os.path.join(out_dir, 'biomarker_decision_table.xlsx')
    result['decision_df'].to_excel(decision_path, index=False)
    result['decision_df'].to_csv(os.path.join(out_dir, 'biomarker_decision_table.csv'),
                                 index=False, encoding='utf-8-sig')

    combo_summary = []
    for n, (auc, _fpr, _tpr) in sorted(result['combo_results'].items()):
        combo_summary.append({
            'top_n': n,
            'auc': float(auc),
            'perm_pval': result['perm_pvals'].get(n),
        })

    table = result['decision_df'].replace([np.inf, -np.inf], np.nan).where(result['decision_df'].notna(), None)
    return jsonify({
        'status': 'ok',
        'tissue': tissue,
        'n_samples': result['n_samples'],
        'n_metabolites': result['n_metabolites'],
        'n_cancer_px': result['n_cancer_px'],
        'n_para_px': result['n_para_px'],
        'combo_summary': combo_summary,
        'roc_img': result['roc_img'],
        'auc_img': result['auc_img'],
        'perm_imgs': result['perm_imgs'],
        'table': table.head(50).to_dict(orient='records'),
        'export_ready': True,
        'warning': '' if result['n_samples'] >= 2 else '当前只有 1 个样本：已输出单代谢物 AUC，RF LOSO 需要至少 2 个样本。',
    })


@bp.route('/api/export', methods=['POST'])
def export():
    ensure_active_batch()
    batch_dir = get_batch_dir()
    body = request.get_json(silent=True) or {}
    tissue = (body.get('tissue') or '').strip()
    if not batch_dir or not tissue:
        return jsonify({'status': 'error', 'message': '缺少 tissue'}), 400
    path = os.path.join(batch_dir, 'classifier', _safe_name(tissue), 'biomarker_decision_table.xlsx')
    if not os.path.exists(path):
        return jsonify({'status': 'error', 'message': '请先运行分类器'}), 404
    return send_file(path, as_attachment=True, download_name=f'biomarker_{_safe_name(tissue)}.xlsx')

