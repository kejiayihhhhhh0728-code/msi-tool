"""
模块 0：样本批次管理
输入：用户在 UI 上的 CRUD 操作
输出：维护 batches/<batch_id>/batch_meta.json 和样本子目录

API 总览
--------
GET  /batch/                         样本管理页
GET  /batch/api/state                返回当前批次 + 样本列表（页面初始化用）
POST /batch/api/new_batch            新建批次（不切样本）
POST /batch/api/switch_batch         切换到已有批次
POST /batch/api/delete_batch         删除批次（连同其下所有样本数据）
POST /batch/api/add_sample           在当前批次内添加样本
POST /batch/api/delete_sample        删除样本
POST /batch/api/switch_sample        切换当前编辑的样本
POST /batch/api/update_sample        修改样本名/分组

设计权衡
- 单用户工具，所有操作直接 read-modify-write batch_meta.json，不做并发锁
- 删除操作直接 shutil.rmtree，UI 层做二次确认
"""
from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime
from typing import Optional

from flask import Blueprint, render_template, request, jsonify, session

from config import BATCH_FOLDER, SAMPLE_SOFT_LIMIT
from blueprints.utils import (
    STAGES,
    create_batch, list_batches, load_batch_meta, save_batch_meta,
    ensure_active_batch, _new_sample_id, _batch_dir,
)

logger = logging.getLogger(__name__)

bp = Blueprint('batch', __name__, url_prefix='/batch')


# ─── 页面 ───────────────────────────────────────────────────────────

@bp.route('/')
def index():
    return render_template('batch.html', active='batch')


# ─── API：状态查询 ──────────────────────────────────────────────────

@bp.route('/api/state')
def state():
    """返回当前 session 的批次状态 + 全部已有批次列表 + 当前批次内样本列表。"""
    bid = ensure_active_batch()
    meta = load_batch_meta(bid)
    return jsonify({
        'status'             : 'ok',
        'current_batch_id'   : bid,
        'current_batch_name' : meta.get('name', ''),
        'current_sample_id'  : meta.get('current_sample_id'),
        'samples'            : meta.get('samples', []),
        'db_uploaded'        : meta.get('db_uploaded', False),
        'all_batches'        : list_batches(),
        'sample_soft_limit'  : SAMPLE_SOFT_LIMIT,
        'stages'             : list(STAGES),
    })


# ─── API：批次操作 ──────────────────────────────────────────────────

@bp.route('/api/new_batch', methods=['POST'])
def new_batch():
    body = request.get_json(silent=True) or {}
    name = (body.get('name') or '').strip() or None
    bid = create_batch(name)
    session['batch_id'] = bid
    session.pop('current_sample_id', None)
    return jsonify({'status': 'ok', 'batch_id': bid})


@bp.route('/api/switch_batch', methods=['POST'])
def switch_batch():
    body = request.get_json(silent=True) or {}
    bid = body.get('batch_id')
    if not bid:
        return jsonify({'status': 'error', 'message': '缺少 batch_id'}), 400
    if not os.path.exists(os.path.join(_batch_dir(bid), 'batch_meta.json')):
        return jsonify({'status': 'error', 'message': '该批次不存在'}), 404
    session['batch_id'] = bid
    meta = load_batch_meta(bid)
    session['current_sample_id'] = meta.get('current_sample_id')
    return jsonify({'status': 'ok'})


@bp.route('/api/delete_batch', methods=['POST'])
def delete_batch():
    body = request.get_json(silent=True) or {}
    bid = body.get('batch_id')
    if not bid:
        return jsonify({'status': 'error', 'message': '缺少 batch_id'}), 400
    bdir = _batch_dir(bid)
    if not os.path.exists(bdir):
        return jsonify({'status': 'error', 'message': '批次不存在'}), 404
    try:
        shutil.rmtree(bdir)
    except OSError as e:
        return jsonify({'status': 'error', 'message': f'删除失败: {e}'}), 500
    # 如果删的是当前活跃批次，清空 session 让 ensure_active_batch 重新挑一个
    if session.get('batch_id') == bid:
        session.pop('batch_id', None)
        session.pop('current_sample_id', None)
        ensure_active_batch()
    return jsonify({'status': 'ok'})


# ─── API：样本操作 ──────────────────────────────────────────────────

@bp.route('/api/add_sample', methods=['POST'])
def add_sample():
    body = request.get_json(silent=True) or {}
    name = (body.get('name') or '').strip()
    group = (body.get('group') or '').strip()
    bid = ensure_active_batch()
    meta = load_batch_meta(bid)

    sid = _new_sample_id()
    sd = os.path.join(_batch_dir(bid), 'samples', sid)
    os.makedirs(sd, exist_ok=True)

    # 自动起名 "样本 N"
    if not name:
        name = f'样本 {len(meta["samples"]) + 1}'

    sample_meta = {
        'id'          : sid,
        'name'        : name,
        'group'       : group,
        'stages_done' : [],
        'msi_filename': '',
        'created_at'  : datetime.now().isoformat(timespec='seconds'),
    }
    import json
    with open(os.path.join(sd, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(sample_meta, f, ensure_ascii=False, indent=2)

    meta['samples'].append({
        'id': sid, 'name': name, 'group': group, 'stages_done': [],
    })
    # 第一个样本自动设为 current
    if not meta.get('current_sample_id'):
        meta['current_sample_id'] = sid
        session['current_sample_id'] = sid
    save_batch_meta(bid, meta)
    return jsonify({'status': 'ok', 'sample_id': sid})


@bp.route('/api/delete_sample', methods=['POST'])
def delete_sample():
    body = request.get_json(silent=True) or {}
    sid = body.get('sample_id')
    if not sid:
        return jsonify({'status': 'error', 'message': '缺少 sample_id'}), 400
    bid = ensure_active_batch()
    meta = load_batch_meta(bid)

    sd = os.path.join(_batch_dir(bid), 'samples', sid)
    if os.path.exists(sd):
        try:
            shutil.rmtree(sd)
        except OSError as e:
            return jsonify({'status': 'error', 'message': f'删除失败: {e}'}), 500

    meta['samples'] = [s for s in meta.get('samples', []) if s['id'] != sid]
    if meta.get('current_sample_id') == sid:
        # 自动切到第一个剩下的
        meta['current_sample_id'] = meta['samples'][0]['id'] if meta['samples'] else None
        session['current_sample_id'] = meta['current_sample_id']
    save_batch_meta(bid, meta)
    return jsonify({'status': 'ok'})


@bp.route('/api/switch_sample', methods=['POST'])
def switch_sample():
    body = request.get_json(silent=True) or {}
    sid = body.get('sample_id')
    if not sid:
        return jsonify({'status': 'error', 'message': '缺少 sample_id'}), 400
    bid = ensure_active_batch()
    meta = load_batch_meta(bid)
    if not any(s['id'] == sid for s in meta.get('samples', [])):
        return jsonify({'status': 'error', 'message': '该样本不在当前批次内'}), 404
    meta['current_sample_id'] = sid
    save_batch_meta(bid, meta)
    session['current_sample_id'] = sid
    return jsonify({'status': 'ok'})


@bp.route('/api/update_sample', methods=['POST'])
def update_sample():
    """修改样本名或分组。"""
    body = request.get_json(silent=True) or {}
    sid = body.get('sample_id')
    if not sid:
        return jsonify({'status': 'error', 'message': '缺少 sample_id'}), 400
    name = body.get('name')
    group = body.get('group')

    bid = ensure_active_batch()
    meta = load_batch_meta(bid)

    target = None
    for s in meta.get('samples', []):
        if s['id'] == sid:
            target = s
            break
    if not target:
        return jsonify({'status': 'error', 'message': '样本不存在'}), 404

    if name is not None:
        target['name'] = str(name).strip() or target['name']
    if group is not None:
        target['group'] = str(group).strip()
    save_batch_meta(bid, meta)

    # 同步样本自身的 meta.json
    sd = os.path.join(_batch_dir(bid), 'samples', sid)
    sample_meta_path = os.path.join(sd, 'meta.json')
    if os.path.exists(sample_meta_path):
        import json
        try:
            with open(sample_meta_path, 'r', encoding='utf-8') as f:
                sm = json.load(f)
            if name is not None:
                sm['name'] = target['name']
            if group is not None:
                sm['group'] = target['group']
            tmp = sample_meta_path + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(sm, f, ensure_ascii=False, indent=2)
            os.replace(tmp, sample_meta_path)
        except (FileNotFoundError, ValueError):
            pass

    return jsonify({'status': 'ok'})
