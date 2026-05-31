"""
PathoSpace-Met：病理组织空间代谢组学全流程分析平台
==================================================
启动方式: python app.py
访问: http://localhost:5000
"""

import logging
import os
import shutil
import time

import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免 GUI 弹窗，必须在 import pyplot 之前设置

from flask import Flask, redirect, url_for, request

from config import (
    BATCH_FOLDER, TMP_FOLDER,
    MAX_CONTENT_LENGTH, SECRET_KEY, TMP_MAX_AGE_DAYS,
)
from blueprints.utils import ensure_active_batch

logger = logging.getLogger(__name__)


# ─── 启动时清理：仅清 tmp/，batches/ 永远不动 ───────────────────────────

def _newest_mtime(path: str) -> float:
    """递归取目录下最新的 mtime；空目录回退到自身 mtime。"""
    newest = os.path.getmtime(path)
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                m = os.path.getmtime(os.path.join(root, name))
                if m > newest:
                    newest = m
            except OSError:
                continue
    return newest


def _cleanup_tmp():
    """启动时清理 tmp/ 下超过 TMP_MAX_AGE_DAYS 天未访问的文件夹。"""
    if not os.path.isdir(TMP_FOLDER):
        return
    cutoff = time.time() - TMP_MAX_AGE_DAYS * 86400
    for name in os.listdir(TMP_FOLDER):
        path = os.path.join(TMP_FOLDER, name)
        if not os.path.isdir(path):
            continue
        try:
            if _newest_mtime(path) < cutoff:
                shutil.rmtree(path)
        except OSError as e:
            logger.warning('cleanup failed: %s (%s)', path, e)


# ─── 注册蓝图 ───────────────────────────────────────────

from blueprints.batch import bp as batch_bp
from blueprints.data_import import bp as data_import_bp
from blueprints.clustering import bp as clustering_bp
from blueprints.nmf_patterns import bp as nmf_bp
from blueprints.rigid_reg import bp as rigid_reg_bp
from blueprints.tps_reg import bp as tps_reg_bp
from blueprints.roi_extract import bp as roi_bp
from blueprints.diff_metabolites import bp as diff_bp
from blueprints.rf_classifier import bp as rf_bp
from blueprints.spatial_heatmap import bp as heatmap_bp
from blueprints.pathway import bp as pathway_bp
from blueprints.annotation_route import bp as annotation_bp

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.secret_key = SECRET_KEY

app.register_blueprint(batch_bp)
app.register_blueprint(data_import_bp)
app.register_blueprint(clustering_bp)
app.register_blueprint(nmf_bp)
app.register_blueprint(rigid_reg_bp)
app.register_blueprint(tps_reg_bp)
app.register_blueprint(roi_bp)
app.register_blueprint(diff_bp)
app.register_blueprint(rf_bp)
app.register_blueprint(heatmap_bp)
app.register_blueprint(pathway_bp)
app.register_blueprint(annotation_bp)


# ─── 全局：确保每个请求都有活跃批次 ─────────────────────────────────

@app.before_request
def _bootstrap_batch():
    """
    保证 session 中有 batch_id 和 current_sample_id。
    没有时自动新建批次和默认空样本（首次访问的体验）。
    仅对页面/API 请求生效，static 资源跳过。
    """
    if request.endpoint == 'static':
        return
    try:
        ensure_active_batch()
    except Exception:
        logger.exception('failed to ensure active batch')


@app.route('/')
def index():
    """根路径跳转到样本管理页（让用户先看到当前批次状态）。"""
    return redirect(url_for('batch.index'))


# ─── 启动 ───────────────────────────────────────────

if __name__ == '__main__':
    _cleanup_tmp()
    print('=' * 55)
    print('  PathoSpace-Met  病理组织空间代谢组学全流程分析平台')
    print('  访问 http://localhost:5000')
    print('=' * 55)
    _debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(debug=_debug, port=5000, use_reloader=_debug)
