"""
模块2：代谢物注释
输入：
  - 当前样本归一化后的 h5ad（来自 sample_dir）
  - 数据库 CSV（用户通过本页面上传到批次根目录，全批次共享）
输出：候选注释表存到 sample_dir/annotation_results.csv

安全设计：
  - 仅接受 multipart 文件上传，不接受用户指定的任意磁盘路径
  - 扩展名白名单：.csv
  - 单文件大小上限 50MB
  - 落盘文件名固定为 hmdb_db.csv（防止路径注入和文件名特殊字符）
  - 列头校验：必须包含 id / name / formula / exact_mass
  - 文件落入批次根目录（batches/<batch_id>/hmdb_db.csv），与其他批次隔离

存储位置（阶段 B 迁移后）：
  - DB CSV       : <batch_dir>/hmdb_db.csv         （每批次一份，跨样本共享）
  - 注释结果 CSV : <sample_dir>/annotation_results.csv  （每个样本独立）
"""
import logging
import os

import pandas as pd
from flask import Blueprint, render_template, request, jsonify, session, send_file
from werkzeug.utils import secure_filename

logger = logging.getLogger(__name__)

from core.annotation_afadesi import annotate_h5ad
from blueprints.utils import (
    get_h5ad_path, get_batch_dir, get_sample_dir,
    load_batch_meta, save_batch_meta, mark_stage_done,
    ensure_active_batch,
)

bp = Blueprint('annotation', __name__, url_prefix='/annotation')

# ─── 安全策略常量 ──────────────────────────────────────────────────────
ALLOWED_DB_EXT = {'.csv'}
MAX_DB_SIZE = 50 * 1024 * 1024              # 50MB
DB_FIXED_NAME = 'hmdb_db.csv'               # 强制重命名，防止文件名注入
REQUIRED_DB_COLUMNS = {'id', 'name', 'formula', 'exact_mass'}


def _get_db_path() -> str | None:
    """在当前批次根目录下查找数据库文件，不接受用户指定的任意路径。"""
    bdir = get_batch_dir()
    if not bdir:
        return None
    local = os.path.join(bdir, DB_FIXED_NAME)
    return local if os.path.exists(local) else None


# ─── 页面 ──────────────────────────────────────────────────────────────

@bp.route('/')
def index():
    return render_template('annotation.html', active='annotation')


# ─── API：上传数据库 CSV ───────────────────────────────────────────────

@bp.route('/api/upload_db', methods=['POST'])
def upload_db():
    """上传代谢物数据库 CSV 到当前批次根目录（与同批次所有样本共享）。"""
    bid = ensure_active_batch()
    bdir = get_batch_dir()
    if not bdir:
        return jsonify({'status': 'error', 'message': '没有活跃批次，请到「样本管理」'}), 400

    f = request.files.get('db_file')
    if not f or not f.filename:
        return jsonify({'status': 'error', 'message': '请选择要上传的 CSV 文件'}), 400

    # 扩展名白名单
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED_DB_EXT:
        return jsonify({'status': 'error', 'message': '只支持 .csv 格式'}), 400

    display_name = secure_filename(f.filename) or 'uploaded.csv'

    if f.content_length and f.content_length > MAX_DB_SIZE:
        mb = MAX_DB_SIZE // 1024 // 1024
        return jsonify({'status': 'error', 'message': f'文件超过 {mb} MB 上限'}), 400

    os.makedirs(bdir, exist_ok=True)
    tmp_path = os.path.join(bdir, '.db_upload.tmp')
    final_path = os.path.join(bdir, DB_FIXED_NAME)
    try:
        f.save(tmp_path)

        size = os.path.getsize(tmp_path)
        if size > MAX_DB_SIZE:
            os.remove(tmp_path)
            mb = MAX_DB_SIZE // 1024 // 1024
            return jsonify({'status': 'error', 'message': f'文件超过 {mb} MB 上限'}), 400

        try:
            head = pd.read_csv(tmp_path, nrows=0, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                head = pd.read_csv(tmp_path, nrows=0, encoding='gbk')
            except Exception as e:
                os.remove(tmp_path)
                return jsonify({'status': 'error', 'message': f'CSV 编码不识别（仅支持 utf-8/gbk）：{e}'}), 400
        except Exception as e:
            os.remove(tmp_path)
            return jsonify({'status': 'error', 'message': f'CSV 解析失败：{e}'}), 400

        cols = set(head.columns)
        miss = REQUIRED_DB_COLUMNS - cols
        if miss:
            os.remove(tmp_path)
            return jsonify({
                'status': 'error',
                'message': f'CSV 缺少必要列：{sorted(miss)}（必需：id, name, formula, exact_mass）',
            }), 400

        os.replace(tmp_path, final_path)

        # 标记批次已上传 DB
        try:
            meta = load_batch_meta(bid)
            meta['db_uploaded'] = True
            save_batch_meta(bid, meta)
        except Exception:
            pass

        return jsonify({
            'status': 'success',
            'message': '数据库上传成功',
            'name': DB_FIXED_NAME,
            'size': size,
            'original_name': display_name,
        })
    except Exception as e:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        logger.exception('upload_db error')
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ─── API：查询数据库上传状态 ───────────────────────────────────────────

@bp.route('/api/db_status')
def db_status():
    """返回当前批次的数据库上传状态，用于页面刷新后恢复显示。"""
    p = _get_db_path()
    if not p:
        return jsonify({'uploaded': False})
    try:
        size = os.path.getsize(p)
    except OSError:
        return jsonify({'uploaded': False})
    return jsonify({
        'uploaded': True,
        'name': DB_FIXED_NAME,
        'size': size,
    })


# ─── API：执行注释 ─────────────────────────────────────────────────────

@bp.route('/api/run', methods=['POST'])
def run_annotation():
    h5ad_path = get_h5ad_path()
    if not h5ad_path:
        return jsonify({'status': 'error', 'message': '当前样本尚未上传或归一化数据'}), 400

    db_csv = _get_db_path()
    if not db_csv:
        return jsonify({'status': 'error', 'message': '请先在本页面上传代谢物数据库 CSV'}), 400

    body = request.get_json(silent=True) or {}
    try:
        mode = body.get('mode', None)
        adduct_names = body.get('adducts', None)

        if isinstance(adduct_names, str):
            adduct_names = [x.strip() for x in adduct_names.split(',') if x.strip()]
        if adduct_names:
            adduct_names = list(adduct_names)
        else:
            adduct_names = None

        if not mode and not adduct_names:
            mode = 'positive'

        out = annotate_h5ad(
            h5ad_path=h5ad_path,
            db_csv_path=db_csv,
            mode=mode,
            adduct_names=adduct_names,
            ppm_tolerance=float(body.get('ppm_tolerance', 10.0)),
            top_n=int(body.get('top_n', 5)),
        )

        sample_dir = get_sample_dir()
        csv_available = False
        preview = []
        if not out['results_df'].empty:
            preview = out['results_df'].head(200).to_dict(orient='records')

        if sample_dir:
            csv_path = os.path.join(sample_dir, 'annotation_results.csv')
            try:
                out['results_df'].to_csv(csv_path, index=False)
                # 写到 session 供模块6 ROI 提取读取注释表（沿用旧 key 名以最小化破坏）
                session['annotation_csv'] = csv_path
                csv_available = True
                mark_stage_done('annotation')
            except Exception:
                logger.exception('failed to save annotation_results.csv')
                csv_available = False

        return jsonify({
            'status': 'success',
            'info': out['info'],
            'preview': preview,
            'csv_available': csv_available,
        })
    except Exception as e:
        logger.exception('annotation error')
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ─── API：下载注释结果 CSV ─────────────────────────────────────────────

@bp.route('/api/download_csv')
def download_csv():
    sample_dir = get_sample_dir()
    if not sample_dir:
        return jsonify({'status': 'error', 'message': '没有活跃样本'}), 404
    csv_path = os.path.join(sample_dir, 'annotation_results.csv')
    if not os.path.exists(csv_path):
        return jsonify({'status': 'error', 'message': '请先执行注释'}), 404
    return send_file(csv_path, as_attachment=True,
                     download_name='annotation_results.csv',
                     mimetype='text/csv')
