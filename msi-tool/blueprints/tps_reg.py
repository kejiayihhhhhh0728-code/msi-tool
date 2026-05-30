"""
模块6：TPS 精细配准
输入：当前样本 reg/ 下的刚性配准结果 + 用户标注的控制点对
输出：TPS 叠加图 + src/dst 控制点（写回 rigid_result.json，供模块7 ROI 读）
"""
import json
import logging
import os

from flask import Blueprint, render_template, request, jsonify, session

from blueprints.utils import get_sample_dir, img_to_b64, mark_stage_done

bp = Blueprint('tps_reg', __name__, url_prefix='/tps-reg')
logger = logging.getLogger(__name__)


def _rigid_json_path() -> str | None:
    """优先样本目录里的 reg/rigid_result.json，找不到回退 session 旧路径。"""
    sd = get_sample_dir()
    if sd:
        p = os.path.join(sd, 'reg', 'rigid_result.json')
        if os.path.exists(p):
            return p
    sess_p = session.get('rigid_result_json')
    if sess_p and os.path.exists(sess_p):
        return sess_p
    return None


def _load_rigid() -> dict | None:
    p = _rigid_json_path()
    if not p:
        return None
    with open(p, encoding='utf-8') as f:
        return json.load(f)


# ─── 页面 ─────────────────────────────────────────────────────────────────────

@bp.route('/')
def index():
    return render_template('tps_reg.html', active='tps_reg')


# ─── API：从 session 获取刚性配准结果图像（供 canvas 加载）────────────────────

@bp.route('/api/session_info')
def session_info():
    d = _load_rigid()
    if not d:
        return jsonify({'status': 'none', 'message': '请先完成刚性配准（模块5）'})
    return jsonify({
        'status':  'ok',
        'tic_img': img_to_b64(d.get('tic_warped_path')),
        'he_img':  img_to_b64(d.get('he_display_path')),
    })


# ─── API：执行 TPS 精配准 ──────────────────────────────────────────────────────

@bp.route('/api/run', methods=['POST'])
def run():
    d = _load_rigid()
    if not d:
        return jsonify({'status': 'error', 'message': '请先完成刚性配准（模块5）'}), 400

    body  = request.get_json(silent=True) or {}
    pairs = body.get('pairs', [])
    if len(pairs) < 3:
        return jsonify({'status': 'error', 'message': '至少需要 3 对控制点'}), 400

    sd = get_sample_dir()
    if not sd:
        return jsonify({'status': 'error', 'message': '没有活跃样本'}), 400
    result_dir = os.path.join(sd, 'reg')
    os.makedirs(result_dir, exist_ok=True)

    try:
        from core.tps_pipeline import run_tps_registration
        res = run_tps_registration(result_dir, result_dir, pairs, d['affine_matrix'])
    except Exception as e:
        logger.exception('tps registration error')
        return jsonify({'status': 'error', 'message': str(e)}), 500

    # 将 TPS 结果写回 rigid_result.json（模块7 ROI 读这里）
    # src/dst_points 必须保存，ROI 提取要做 forward TPS warp
    json_path = _rigid_json_path()
    if json_path and os.path.exists(json_path):
        d['tps_overlay_path']    = res['overlay_path']
        d['tps_trace_path']      = res.get('coordinate_trace_path', '')
        d['tps_src_points']      = res.get('src_points', [])
        d['tps_dst_points']      = res.get('dst_points', [])
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(d, f)
        session['rigid_result_json'] = json_path
        mark_stage_done('tps')

    return jsonify({
        'status':      'success',
        'metrics':     res['metrics'],
        'overlay_img': img_to_b64(res['overlay_path']),
        'n_pairs':     len(pairs),
    })
