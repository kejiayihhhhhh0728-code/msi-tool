"""模块8：差异代谢物筛选。"""
from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Blueprint, render_template, request, jsonify, send_file

from blueprints.utils import ensure_active_batch, get_batch_dir
from core.cohort import assemble_cohort, cohort_status
from core.differential import run_full_pipeline, run_single_sample_pixel_pipeline, DEFAULT_FC_CUTOFF

bp = Blueprint('diff_metabolites', __name__, url_prefix='/diff')


@bp.route('/')
def index():
    return render_template('diff.html', active='diff_metabolites')


@bp.route('/api/status')
def status():
    """返回当前批次中模块8可用的 tissue 和样本配对状态。"""
    ensure_active_batch()
    batch_dir = get_batch_dir()
    if not batch_dir:
        return jsonify({'status': 'error', 'message': '未找到当前批次'}), 400

    stat = cohort_status(batch_dir)
    cohorts = assemble_cohort(batch_dir)
    tissues = []
    for tissue_name, data in cohorts.items():
        tissues.append({
            'name': tissue_name,
            'n_samples': data['n_samples'],
            'n_metabolites': int(data['cancer_mat'].shape[0]),
            'sample_names': data.get('sample_names', []),
        })
    tissues.sort(key=lambda x: x['name'])
    return jsonify({'status': 'ok', 'cohort_status': stat, 'tissues': tissues})


@bp.route('/api/run', methods=['POST'])
def run():
    """执行三重筛选（paired test + log2FC + VIP）。"""
    ensure_active_batch()
    batch_dir = get_batch_dir()
    if not batch_dir:
        return jsonify({'status': 'error', 'message': '未找到当前批次'}), 400

    body = request.get_json(silent=True) or {}
    tissue_name = (body.get('tissue') or '').strip()
    p_cutoff = _as_float(body.get('p_cutoff'), 0.05)
    fc_cutoff = _as_float(body.get('fc_cutoff'), DEFAULT_FC_CUTOFF)
    vip_cutoff = _as_float(body.get('vip_cutoff'), 1.0)
    pls_perm = int(_as_float(body.get('plsda_n_perm'), 200))
    do_median = bool(body.get('do_median_correction', True))
    do_pca = bool(body.get('do_pca', True))
    do_plsda = bool(body.get('do_plsda', True))

    cohorts = assemble_cohort(batch_dir)
    if not cohorts:
        return jsonify({
            'status': 'error',
            'message': '当前批次没有可配对的 cancer/paracancer ROI 数据。请先让至少 1 个样本完成 ROI 提取。'
        }), 400

    if not tissue_name:
        tissue_name = sorted(cohorts.keys())[0]
    if tissue_name not in cohorts:
        return jsonify({'status': 'error', 'message': f'未找到组织: {tissue_name}'}), 404

    cohort = cohorts[tissue_name]
    if cohort['n_samples'] == 1:
        result = run_single_sample_pixel_pipeline(
            batch_dir,
            tissue_name,
            fc_cutoff=fc_cutoff,
            p_cutoff=p_cutoff,
            vip_cutoff=vip_cutoff,
        )
    else:
        result = run_full_pipeline(
            cohort['cancer_mat'],
            cohort['para_mat'],
            group_name=tissue_name,
            fc_cutoff=fc_cutoff,
            p_cutoff=p_cutoff,
            vip_cutoff=vip_cutoff,
            do_median_correction=do_median,
            do_pca=do_pca,
            do_plsda=do_plsda,
            plsda_n_perm=pls_perm,
            rng_seed=2026,
        )
    if result.get('error'):
        return jsonify({'status': 'error', 'message': result['error']}), 400

    out_dir = os.path.join(batch_dir, 'diff', _safe_name(tissue_name))
    os.makedirs(out_dir, exist_ok=True)
    diff_df = result['diff_df'].copy()
    diff_df.to_csv(os.path.join(out_dir, 'differential_metabolites.csv'), index=False, encoding='utf-8-sig')
    excel_path = os.path.join(out_dir, 'differential_metabolites.xlsx')
    _write_excel(excel_path, diff_df, result, cohort, {
        'tissue': tissue_name,
        'p_cutoff': p_cutoff,
        'fc_cutoff': fc_cutoff,
        'vip_cutoff': vip_cutoff,
        'plsda_n_perm': pls_perm,
        'do_median_correction': do_median,
    })

    final_hits = diff_df[diff_df['final_change'] != 'Non'].copy()
    table = final_hits.sort_values(
        ['FDR', 'mean_log2FC'], ascending=[True, False]
    ).head(50)
    if table.empty:
        table = diff_df.sort_values('FDR').head(50)

    warning = result.get('warning') or ''
    if not warning and cohort['n_samples'] < 6:
        warning = '配对样本数 < 6，已按参考代码自动使用配对 t 检验；n >= 6 时使用配对 Wilcoxon。'

    return jsonify({
        'status': 'ok',
        'tissue': tissue_name,
        'n_samples': cohort['n_samples'],
        'n_metabolites': int(cohort['cancer_mat'].shape[0]),
        'sample_names': cohort.get('sample_names', []),
        'test_used': result['test_used'],
        'warning': warning,
        'n_up': result['n_up'],
        'n_down': result['n_down'],
        'n_total': result['n_total'],
        'plsda_r2': _json_float(result.get('plsda_r2')),
        'plsda_cv_auc': _json_float(result.get('plsda_cv_auc')),
        'plsda_q2': _json_float(result.get('plsda_q2')),
        'plsda_perm_pval': _json_float(result.get('plsda_perm_pval')),
        'volcano_img': result.get('volcano_img'),
        'pca_img': result.get('pca_img'),
        'plsda_img': result.get('plsda_img'),
        'plsda_perm_img': result.get('plsda_perm_img'),
        'fc_heatmap_img': result.get('fc_heatmap_img'),
        'table': _df_for_json(table),
        'export_ready': True,
        'export_tissue': tissue_name,
        'updated_at': datetime.now().isoformat(timespec='seconds'),
    })


@bp.route('/api/export', methods=['POST'])
def export():
    """导出最近一次差异分析结果 Excel。"""
    ensure_active_batch()
    batch_dir = get_batch_dir()
    body = request.get_json(silent=True) or {}
    tissue_name = (body.get('tissue') or '').strip()
    if not batch_dir or not tissue_name:
        return jsonify({'status': 'error', 'message': '缺少 tissue'}), 400
    path = os.path.join(batch_dir, 'diff', _safe_name(tissue_name), 'differential_metabolites.xlsx')
    if not os.path.exists(path):
        return jsonify({'status': 'error', 'message': '请先运行差异分析'}), 404
    return send_file(path, as_attachment=True, download_name=f'differential_{_safe_name(tissue_name)}.xlsx')


def _as_float(value, default: float) -> float:
    try:
        v = float(value)
        if np.isfinite(v):
            return v
    except (TypeError, ValueError):
        pass
    return default


def _safe_name(name: str) -> str:
    return ''.join(c if c.isalnum() or c in ('-', '_', '.') else '_' for c in str(name))[:80] or 'tissue'


def _json_float(value):
    if value is None:
        return None
    try:
        v = float(value)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _df_for_json(df: pd.DataFrame) -> list[dict]:
    cols = ['Metabolite', 'mean_log2FC', 'PValue', 'FDR', 'VIP', 'final_change',
            'direction_consistency', 'Pixel_AUC', 'Pixel_AUC_abs', 'Effect_rank_biserial',
            'cancer_grand_mean', 'paracancer_grand_mean']
    keep = [c for c in cols if c in df.columns]
    out = df[keep].replace([np.inf, -np.inf], np.nan).where(pd.notna(df[keep]), None)
    rows = []
    for row in out.to_dict(orient='records'):
        clean = {}
        for k, v in row.items():
            if isinstance(v, float):
                clean[k] = None if not np.isfinite(v) else v
            else:
                clean[k] = v
        rows.append(clean)
    return rows


def _write_excel(path: str, diff_df: pd.DataFrame, result: dict, cohort: dict, params: dict) -> None:
    summary = pd.DataFrame([
        {'key': 'tissue', 'value': params['tissue']},
        {'key': 'paired_samples', 'value': cohort['n_samples']},
        {'key': 'metabolites', 'value': int(cohort['cancer_mat'].shape[0])},
        {'key': 'test_used', 'value': result['test_used']},
        {'key': 'up', 'value': result['n_up']},
        {'key': 'down', 'value': result['n_down']},
        {'key': 'p_cutoff', 'value': params['p_cutoff']},
        {'key': 'fc_cutoff_log2', 'value': params['fc_cutoff']},
        {'key': 'vip_cutoff', 'value': params['vip_cutoff']},
        {'key': 'plsda_r2', 'value': result.get('plsda_r2')},
        {'key': 'plsda_cv_auc', 'value': result.get('plsda_cv_auc')},
        {'key': 'plsda_q2', 'value': result.get('plsda_q2')},
        {'key': 'plsda_perm_pval', 'value': result.get('plsda_perm_pval')},
    ])
    samples = pd.DataFrame({
        'sample_id': cohort.get('sample_ids', []),
        'sample_name': cohort.get('sample_names', []),
        'sample_group': cohort.get('sample_groups', []),
    })
    with pd.ExcelWriter(path) as writer:
        summary.to_excel(writer, sheet_name='summary', index=False)
        samples.to_excel(writer, sheet_name='paired_samples', index=False)
        diff_df.to_excel(writer, sheet_name='differential', index=False)
