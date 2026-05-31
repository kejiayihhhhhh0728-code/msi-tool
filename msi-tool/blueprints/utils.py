"""
蓝图层共享工具函数（不含业务逻辑，仅做 session/批次/样本相关的目录解析与元数据更新）

批次/样本生命周期模型
---------------------
- session['batch_id']           当前活跃批次 ID（首次访问时自动创建）
- session['current_sample_id']  当前正在编辑的样本 ID
- batches/<batch_id>/batch_meta.json   批次的 single source of truth
- batches/<batch_id>/samples/<sample_id>/meta.json   每个样本的元数据

设计权衡（竞赛展示工具）
- 单用户使用，不做并发锁；多标签同时编辑同一批次会互相覆盖 current_sample_id，
  设计上接受这一限制，UI 文案需提示"建议单标签操作"。
- batch_meta.json 采用直接 read-modify-write，单线程下安全。
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime
from typing import Optional

from flask import session
from werkzeug.datastructures import FileStorage

from config import BATCH_FOLDER, UPLOAD_CACHE


# ─── 阶段：每个样本可达到的工作流阶段名 ──────────────────────────────────
STAGES = ('import', 'norm', 'rigid', 'tps', 'annotation', 'roi')


# ─── ID 生成 ───────────────────────────────────────────────────────────

def _new_batch_id() -> str:
    return 'b_' + uuid.uuid4().hex[:8]


def _new_sample_id() -> str:
    return 's_' + uuid.uuid4().hex[:8]


# ─── 批次：创建、加载、更新 ─────────────────────────────────────────────

def _batch_dir(batch_id: str) -> str:
    return os.path.join(BATCH_FOLDER, batch_id)


def _batch_meta_path(batch_id: str) -> str:
    return os.path.join(_batch_dir(batch_id), 'batch_meta.json')


def load_batch_meta(batch_id: str) -> dict:
    """读取批次元数据；不存在则抛 FileNotFoundError。"""
    with open(_batch_meta_path(batch_id), 'r', encoding='utf-8') as f:
        return json.load(f)


def save_batch_meta(batch_id: str, meta: dict) -> None:
    """原子写回批次元数据。"""
    path = _batch_meta_path(batch_id)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def create_batch(name: Optional[str] = None) -> str:
    """新建一个空批次，返回 batch_id。"""
    bid = _new_batch_id()
    bdir = _batch_dir(bid)
    os.makedirs(os.path.join(bdir, 'samples'), exist_ok=True)
    meta = {
        'batch_id'         : bid,
        'name'             : name or f'批次 {datetime.now().strftime("%Y-%m-%d %H:%M")}',
        'created_at'       : datetime.now().isoformat(timespec='seconds'),
        'current_sample_id': None,
        'samples'          : [],          # [{id, name, group, stages_done}]
        'db_uploaded'      : False,
    }
    save_batch_meta(bid, meta)
    return bid


def list_batches() -> list[dict]:
    """列出所有已存在的批次（按创建时间倒序，最近的在前）。"""
    if not os.path.isdir(BATCH_FOLDER):
        return []
    out = []
    for name in os.listdir(BATCH_FOLDER):
        if not name.startswith('b_'):
            continue
        try:
            meta = load_batch_meta(name)
            out.append({
                'batch_id'   : meta['batch_id'],
                'name'       : meta.get('name', name),
                'created_at' : meta.get('created_at', ''),
                'sample_count': len(meta.get('samples', [])),
            })
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    out.sort(key=lambda b: b['created_at'], reverse=True)
    return out


def ensure_active_batch() -> str:
    """
    确保 session 有一个活跃批次：
      - 若 session 已有有效 batch_id，返回它
      - 若没有但磁盘上有批次，使用最近的那个
      - 否则新建一个空批次（同时创建一个默认空样本，用户能立刻进入模块1）
    """
    bid = session.get('batch_id')
    if bid and os.path.exists(_batch_meta_path(bid)):
        return bid

    existing = list_batches()
    if existing:
        bid = existing[0]['batch_id']
    else:
        bid = create_batch()
        # 顺手建一个默认空样本，新用户进入模块1 不用先去样本管理页
        meta = load_batch_meta(bid)
        sid = _new_sample_id()
        sample_dir = os.path.join(_batch_dir(bid), 'samples', sid)
        os.makedirs(sample_dir, exist_ok=True)
        sample_meta = {
            'id'           : sid,
            'name'         : '样本 1',
            'group'        : '',
            'stages_done'  : [],
            'msi_filename' : '',
            'created_at'   : datetime.now().isoformat(timespec='seconds'),
        }
        with open(os.path.join(sample_dir, 'meta.json'), 'w', encoding='utf-8') as f:
            json.dump(sample_meta, f, ensure_ascii=False, indent=2)
        meta['samples'].append({
            'id': sid, 'name': sample_meta['name'], 'group': '', 'stages_done': [],
        })
        meta['current_sample_id'] = sid
        save_batch_meta(bid, meta)

    session['batch_id'] = bid
    # 同步 current_sample_id 到 session
    meta = load_batch_meta(bid)
    if meta.get('current_sample_id'):
        session['current_sample_id'] = meta['current_sample_id']
    return bid


# ─── 样本目录解析 ──────────────────────────────────────────────────────

def get_current_sample_id() -> Optional[str]:
    """读 session 里的当前样本 ID；session 缺失时回退到 batch_meta 中的指针。"""
    sid = session.get('current_sample_id')
    if sid:
        return sid
    bid = session.get('batch_id')
    if bid:
        try:
            meta = load_batch_meta(bid)
            sid = meta.get('current_sample_id')
            if sid:
                session['current_sample_id'] = sid
                return sid
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    return None


def get_sample_dir(sample_id: Optional[str] = None) -> Optional[str]:
    """
    返回某个样本的工作目录绝对路径；目录已确保存在。
    sample_id 省略时使用当前样本。无活跃样本时返回 None。
    """
    bid = session.get('batch_id')
    sid = sample_id or get_current_sample_id()
    if not bid or not sid:
        return None
    d = os.path.join(_batch_dir(bid), 'samples', sid)
    os.makedirs(d, exist_ok=True)
    return d


def get_batch_dir() -> Optional[str]:
    """返回当前批次目录，无活跃批次时返回 None。"""
    bid = session.get('batch_id')
    return _batch_dir(bid) if bid else None


# ─── 样本元数据更新 ────────────────────────────────────────────────────

def load_sample_meta(sample_id: Optional[str] = None) -> Optional[dict]:
    sd = get_sample_dir(sample_id)
    if not sd:
        return None
    p = os.path.join(sd, 'meta.json')
    if not os.path.exists(p):
        return None
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_sample_meta(meta: dict, sample_id: Optional[str] = None) -> None:
    sd = get_sample_dir(sample_id)
    if not sd:
        raise RuntimeError('No active sample to save meta for.')
    p = os.path.join(sd, 'meta.json')
    tmp = p + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)


def mark_stage_done(stage: str, sample_id: Optional[str] = None) -> None:
    """
    标记某个样本完成了某个工作流阶段，同时更新 batch_meta 里的索引。
    stage 必须是 STAGES 之一；非法值会静默忽略。
    """
    if stage not in STAGES:
        return
    meta = load_sample_meta(sample_id)
    if not meta:
        return
    if stage not in meta['stages_done']:
        meta['stages_done'].append(stage)
        save_sample_meta(meta, sample_id)

    # 同步到 batch_meta 里对应样本条目
    bid = session.get('batch_id')
    sid = sample_id or get_current_sample_id()
    if not bid or not sid:
        return
    try:
        bmeta = load_batch_meta(bid)
        for s in bmeta.get('samples', []):
            if s['id'] == sid:
                if stage not in s.get('stages_done', []):
                    s.setdefault('stages_done', []).append(stage)
                break
        save_batch_meta(bid, bmeta)
    except (FileNotFoundError, json.JSONDecodeError):
        pass


# ─── SHA-256 上传缓存 ──────────────────────────────────────────────────

def cache_upload(file: FileStorage, ext_hint: Optional[str] = None) -> tuple[str, str]:
    """
    把上传文件写入 library/upload_cache/<sha16>.<ext>，跨批次去重。

    参数
    ----
    file       : werkzeug FileStorage（来自 request.files）
    ext_hint   : 可选扩展名（含点，如 '.txt'）；省略时从原始 filename 取

    返回
    ----
    (sha16, cache_path)
       sha16     : SHA-256 前 16 位（hex），可作为该文件的指纹
       cache_path: 绝对路径（保证存在）

    实现要点
    ----
    - 先把流写入临时文件并同步算 hash（不一次性读入内存，支持大文件）
    - hash 已存在 → 删除临时文件，复用缓存；否则原子改名
    """
    os.makedirs(UPLOAD_CACHE, exist_ok=True)

    if ext_hint:
        ext = ext_hint
    else:
        ext = os.path.splitext(file.filename or '')[1].lower() or '.bin'

    # 写临时文件 + 流式 hash
    tmp_name = f'.upload-{uuid.uuid4().hex[:8]}.tmp'
    tmp_path = os.path.join(UPLOAD_CACHE, tmp_name)
    h = hashlib.sha256()
    file.stream.seek(0)
    with open(tmp_path, 'wb') as f:
        while True:
            chunk = file.stream.read(1 << 20)  # 1MB
            if not chunk:
                break
            h.update(chunk)
            f.write(chunk)
    file.stream.seek(0)  # 还原游标，方便调用方再用

    sha16 = h.hexdigest()[:16]
    cache_path = os.path.join(UPLOAD_CACHE, sha16 + ext)
    if os.path.exists(cache_path):
        os.remove(tmp_path)
    else:
        os.replace(tmp_path, cache_path)
    return sha16, cache_path


def materialize_into_sample(cache_path: str, sample_dir: str, dest_name: str) -> str:
    """
    把缓存里的文件以 dest_name 暴露到样本目录。

    Windows 上 hardlink 仅同盘内可用，symlink 需要管理员权限。最稳妥的兜底是
    复制；考虑到 MSI 单文件几百 MB，先尝试 hardlink，失败再 fallback 到 copy。
    """
    dest = os.path.join(sample_dir, dest_name)
    if os.path.exists(dest):
        os.remove(dest)
    try:
        os.link(cache_path, dest)         # 同盘 hardlink，秒级
    except OSError:
        shutil.copyfile(cache_path, dest)  # 跨盘/无权限 fallback
    return dest


# ─── 旧接口保留：模块迁移期间的过渡函数 ─────────────────────────────────

def get_h5ad_path() -> Optional[str]:
    """
    从 session/样本目录中取最近一次归一化后的 h5ad 路径；
    没有归一化结果时退回 raw。

    迁移期：先尝试当前样本目录里的固定文件名，其次回退到旧的 session key。
    """
    sd = get_sample_dir()
    if sd:
        for name in ('norm_tic.h5ad', 'norm_rms.h5ad', 'raw.h5ad'):
            p = os.path.join(sd, name)
            if os.path.exists(p):
                return p

    # 旧路径回退（阶段 B 全部模块迁移完成后可删）
    for key in ('h5ad_norm_tic', 'h5ad_norm_rms', 'h5ad_raw'):
        p = session.get(key)
        if p and os.path.exists(p):
            return p
    return None


def img_to_b64(path: str) -> Optional[str]:
    """将图像文件编码为 base64 data URI；path 不存在时返回 None。"""
    if not path or not os.path.exists(path):
        return None
    ext = os.path.splitext(path)[1].lower()
    mime = {
        '.png':  'image/png',
        '.jpg':  'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.tif':  'image/tiff',
        '.tiff': 'image/tiff',
    }.get(ext, 'image/png')
    with open(path, 'rb') as f:
        return f'data:{mime};base64,{base64.b64encode(f.read()).decode()}'
