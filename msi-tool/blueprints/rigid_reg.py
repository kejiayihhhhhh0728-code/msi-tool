"""
模块5：刚性配准
输入：MSI txt + HE 图像（上传到当前样本的 reg/ 子目录）
输出：配准叠加图 + 仿射矩阵（写入 sample_dir/reg/，路径同步进 session 方便其他模块读）
"""
import json
import logging
import os

from flask import Blueprint, render_template, request, jsonify, session
from werkzeug.utils import secure_filename

from blueprints.utils import get_sample_dir, mark_stage_done, img_to_b64

bp = Blueprint('rigid_reg', __name__, url_prefix='/rigid-reg')
logger = logging.getLogger(__name__)

ALLOWED_MSI = {'.txt', '.tsv', '.csv'}
ALLOWED_IMG = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}


def _reg_dir() -> str | None:
    """当前样本下的 reg/ 子目录；无活跃样本返回 None。"""
    sd = get_sample_dir()
    if not sd:
        return None
    d = os.path.join(sd, 'reg')
    os.makedirs(d, exist_ok=True)
    return d


def _existing_msi_path() -> str | None:
    """查找当前样本中由模块1 持久化的 MSI 原始文件，供刚性配准复用。"""
    sd = get_sample_dir()
    if not sd:
        return None
    for ext in ('.txt', '.tsv', '.csv'):
        p = os.path.join(sd, f'msi_raw{ext}')
        if os.path.exists(p):
            return p
    return None


# ─── 页面 ─────────────────────────────────────────────────────────────────────

@bp.route('/')
def index():
    return render_template('rigid_reg.html', active='rigid_reg')


# ─── API：执行刚性配准 ─────────────────────────────────────────────────────────

@bp.route('/api/run', methods=['POST'])
def run():
    msi_f = request.files.get('msi_file')         # 可选：留空则用模块1 已上传的数据
    he_f  = request.files.get('he_file')

    # HE 必传（模块1 不涉及 HE）
    if not he_f or not he_f.filename:
        return jsonify({'status': 'error', 'message': '请上传 HE 图像'}), 400
    if os.path.splitext(he_f.filename)[1].lower() not in ALLOWED_IMG:
        return jsonify({'status': 'error', 'message': 'HE 图像支持 .png / .jpg / .tif'}), 400

    reg_dir = _reg_dir()
    if not reg_dir:
        return jsonify({'status': 'error', 'message': '没有活跃样本，请到「样本管理」'}), 400

    # MSI: 优先用本次上传的；否则复用模块1 落到样本目录的 msi_raw.<ext>
    if msi_f and msi_f.filename:
        if os.path.splitext(msi_f.filename)[1].lower() not in ALLOWED_MSI:
            return jsonify({'status': 'error', 'message': 'MSI 文件支持 .txt / .tsv / .csv'}), 400
        msi_path = os.path.join(reg_dir, secure_filename(msi_f.filename))
        msi_f.save(msi_path)
    else:
        msi_path = _existing_msi_path()
        if not msi_path:
            return jsonify({
                'status': 'error',
                'message': '当前样本未找到 MSI 数据，请先在「数据导入」上传，或在此页面手动选择'
            }), 400

    he_path  = os.path.join(reg_dir, secure_filename(he_f.filename))
    he_f.save(he_path)

    try:
        from core.reg_pipeline import run_multimodal_registration
        res = run_multimodal_registration(msi_path, he_path, reg_dir)
    except Exception as e:
        logger.exception('rigid registration error')
        return jsonify({'status': 'error', 'message': str(e)}), 500

    # 持久化结果到 sample_dir/reg/rigid_result.json；session 同步路径以兼容
    result_data = {
        'affine_matrix':   res['affine_matrix'],
        'metrics':         res['metrics'],
        'overlay_path':    res['overlay_path'],
        'tic_warped_path': res['tic_warped_path'],
        'he_display_path': res['he_display_path'],
    }
    json_path = os.path.join(reg_dir, 'rigid_result.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f)
    session['rigid_result_json'] = json_path
    mark_stage_done('rigid')

    return jsonify({
        'status':      'success',
        'metrics':     res['metrics'],
        'overlay_img': img_to_b64(res['overlay_path']),
    })


# ─── API：读取已有结果（页面刷新后恢复）─────────────────────────────────────────

@bp.route('/api/result_info')
def result_info():
    """
    回报：
      - 是否有已完成的配准（用于刷新页面后恢复结果）
      - 当前样本是否已经有可复用的 MSI 文件（用于 UI 隐藏/标记 MSI 上传卡）
    """
    # 1) MSI 自动可用性
    auto = _existing_msi_path()
    msi_auto = {
        'available': auto is not None,
        'name'     : os.path.basename(auto) if auto else '',
    }

    # 2) 已有配准结果
    candidates = []
    reg_dir = _reg_dir()
    if reg_dir:
        candidates.append(os.path.join(reg_dir, 'rigid_result.json'))
    sess_p = session.get('rigid_result_json')
    if sess_p:
        candidates.append(sess_p)

    for p in candidates:
        if p and os.path.exists(p):
            with open(p, encoding='utf-8') as f:
                d = json.load(f)
            session['rigid_result_json'] = p
            return jsonify({
                'status':      'ok',
                'metrics':     d.get('metrics'),
                'overlay_img': img_to_b64(d.get('overlay_path')),
                'msi_auto'   : msi_auto,
            })
    return jsonify({'status': 'none', 'msi_auto': msi_auto})
