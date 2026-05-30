"""
模块7：ROI 标注与区域提取
输入：配准后的 HE 叠加图 + MSI 数据（h5ad）+ 配准仿射矩阵
输出：各 ROI 区域的 MSI 像素强度 CSV + 代谢物注释 + pseudo-bulk 矩阵
依赖：模块1（数据导入）、模块2（代谢物注释）、模块5/6（配准，可选）

工作流程
--------
1. 加载配准叠加图（从 session 或用户上传）
2. 用户在 canvas 上绘制多边形 ROI（癌区/癌旁区/自定义）
3. 后端将多边形映射到 MSI 坐标，提取对应像素数据
4. 输出：
   - 各区域像素坐标 + 全m/z强度宽表 CSV
   - 各区域 pseudo-bulk 汇总 CSV
   - 代谢物注释已融合到列名
"""
import logging
import os
import json

import numpy as np

logger = logging.getLogger(__name__)
import pandas as pd
import anndata as ad
from flask import (Blueprint, render_template, request, jsonify,
                   session, current_app, send_file)

from core.pseudobulk import (
    load_affine_matrix, extract_rois,
    build_intensity_dataframe, compute_pseudobulk,
    plot_roi_preview,
)
from blueprints.utils import (
    get_h5ad_path, get_sample_dir, load_sample_meta,
    mark_stage_done, img_to_b64,
)

bp = Blueprint('roi_extract', __name__, url_prefix='/roi')

ALLOWED_IMG = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
ALLOWED_JSON = {'.json'}


# ─── 工具函数 ──────────────────────────────────────────────────────


def _get_annotation_csv_path() -> str | None:
    """注释 CSV 优先样本目录，回退 session 旧 key"""
    sd = get_sample_dir()
    if sd:
        p = os.path.join(sd, 'annotation_results.csv')
        if os.path.exists(p):
            return p
    p = session.get('annotation_csv')
    return p if (p and os.path.exists(p)) else None


def _get_annotation_df() -> pd.DataFrame | None:
    """读取代谢物注释 CSV（模块2产出）"""
    p = _get_annotation_csv_path()
    if p:
        try:
            return pd.read_csv(p)
        except Exception:
            pass
    return None


def _rigid_json_path() -> str | None:
    """rigid_result.json 优先样本目录，回退 session 旧 key"""
    sd = get_sample_dir()
    if sd:
        p = os.path.join(sd, 'reg', 'rigid_result.json')
        if os.path.exists(p):
            return p
    p = session.get('rigid_result_json')
    return p if (p and os.path.exists(p)) else None


def _get_rigid_result() -> dict | None:
    """读取刚性配准结果 JSON"""
    p = _rigid_json_path()
    if not p:
        return None
    try:
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _ensure_roi_dir() -> str | None:
    """ROI 输出目录：当前样本下的 roi/ 子目录"""
    sd = get_sample_dir()
    if not sd:
        return None
    roi_dir = os.path.join(sd, 'roi')
    os.makedirs(roi_dir, exist_ok=True)
    return roi_dir



def _get_img_shape(img_path: str) -> tuple[int, int]:
    """返回图像的 (width, height)（像素）"""
    import cv2
    img = cv2.imdecode(
        np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED
    )
    if img is None:
        raise ValueError(f'无法读取图像: {img_path}')
    h, w = img.shape[:2]
    return w, h


# ─── 页面路由 ──────────────────────────────────────────────────────

@bp.route('/')
def index():
    return render_template('roi.html', active='roi_extract')


# ─── API：查询 session 状态 ────────────────────────────────────────

@bp.route('/api/session_info')
def session_info():
    """返回当前 session 中可用的数据信息"""
    h5ad_path = get_h5ad_path()
    rigid_res  = _get_rigid_result()
    ann_csv    = session.get('annotation_csv')

    # 检查 TPS 数据是否齐备：要求 rigid_result.json 里既有 src 又有 dst，且 >= 3 对
    has_tps = False
    if rigid_res:
        src_pts = rigid_res.get('tps_src_points') or []
        dst_pts = rigid_res.get('tps_dst_points') or []
        has_tps = len(src_pts) >= 3 and len(src_pts) == len(dst_pts)

    # 新流程：从样本 meta 读 MSI 文件名；旧流程兜底用 session
    sample_meta = load_sample_meta() or {}
    info = {
        'has_msi_data'    : h5ad_path is not None,
        'has_registration': rigid_res is not None,
        'has_tps'         : has_tps,
        'has_annotation'  : _get_annotation_csv_path() is not None,
        'msi_filename'    : sample_meta.get('msi_filename') or session.get('msi_filename', ''),
        'overlay_img'     : None,
        'img_w'           : None,
        'img_h'           : None,
    }

    # 优先返回 HE 原图（标注更清晰）；找不到再回退到叠加图
    # 注意：ROI 提取仍用配准矩阵 + 可选 TPS 把多边形映射到 MSI 像素，
    # 显示与计算分离 —— 用户在 HE 上画框，后端在 MSI 上找像素
    if rigid_res:
        candidates = [
            rigid_res.get('he_display_path'),
            rigid_res.get('overlay_path'),
        ]
        for img_p in candidates:
            if img_p and os.path.exists(img_p):
                try:
                    info['overlay_img'] = img_to_b64(img_p)
                    info['img_w'], info['img_h'] = _get_img_shape(img_p)
                    session['roi_overlay_path'] = img_p
                    break
                except Exception:
                    continue

    # 若 session 中有之前保存的图，回退使用
    if not info['overlay_img']:
        saved = session.get('roi_overlay_path')
        if saved and os.path.exists(saved):
            try:
                info['overlay_img'] = img_to_b64(saved)
                info['img_w'], info['img_h'] = _get_img_shape(saved)
            except Exception:
                pass

    return jsonify({'status': 'ok', 'info': info})


@bp.route('/api/saved_state')
def saved_state():
    """读取当前样本已保存的 ROI 多边形和导出结果，用于页面重新进入时恢复。"""
    roi_dir = _ensure_roi_dir()
    if not roi_dir:
        return jsonify({'status': 'ok', 'has_saved': False})

    polygons_path = os.path.join(roi_dir, 'roi_polygons.json')
    polygons_meta = None
    if os.path.exists(polygons_path):
        try:
            with open(polygons_path, 'r', encoding='utf-8') as f:
                polygons_meta = json.load(f)
        except Exception:
            polygons_meta = None

    result_files = []
    if os.path.isdir(roi_dir):
        for name in os.listdir(roi_dir):
            if name.endswith('_intensity.csv') or name == 'pseudobulk_all.csv':
                result_files.append(name)

    roi_stats = []
    pb_csv = os.path.join(roi_dir, 'pseudobulk_all.csv')
    if os.path.exists(pb_csv):
        try:
            pb = pd.read_csv(pb_csv)
            for _, row in pb.iterrows():
                roi_stats.append({
                    'tissue_name' : row.get('tissue_name', ''),
                    'sub_roi_name': row.get('region_name', ''),
                    'type'        : row.get('region_type', ''),
                    'pixel_count' : int(row.get('pixel_count', 0)),
                    'n_mz'        : max(0, len(pb.columns) - 4),
                })
        except Exception:
            roi_stats = []

    return jsonify({
        'status': 'ok',
        'has_saved': bool(polygons_meta or result_files),
        'polygons_meta': polygons_meta,
        'result_files': result_files,
        'roi_stats': roi_stats,
        'pseudobulk_csv': 'pseudobulk_all.csv' if os.path.exists(pb_csv) else None,
    })


# ─── API：上传 HE/叠加图 ───────────────────────────────────────────

@bp.route('/api/upload_image', methods=['POST'])
def upload_image():
    """用户手动上传 HE 图像或配准叠加图"""
    img_f = request.files.get('image_file')
    if not img_f or not img_f.filename:
        return jsonify({'status': 'error', 'message': '请上传图像文件'}), 400

    ext = os.path.splitext(img_f.filename)[1].lower()
    if ext not in ALLOWED_IMG:
        return jsonify({'status': 'error',
                        'message': '支持格式：PNG / JPG / TIF'}), 400

    sd = get_sample_dir()
    if not sd:
        return jsonify({'status': 'error', 'message': '没有活跃样本'}), 400
    roi_dir = os.path.join(sd, 'roi')
    os.makedirs(roi_dir, exist_ok=True)

    img_path = os.path.join(roi_dir, f'roi_overlay{ext}')
    img_f.save(img_path)
    session['roi_overlay_path'] = img_path

    try:
        img_b64 = img_to_b64(img_path)
        img_w, img_h = _get_img_shape(img_path)
        return jsonify({
            'status' : 'success',
            'img_b64': img_b64,
            'img_w'  : img_w,
            'img_h'  : img_h,
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ─── API：上传配准 JSON ────────────────────────────────────────────

@bp.route('/api/upload_reg_json', methods=['POST'])
def upload_reg_json():
    """用户手动上传 rigid_result.json"""
    json_f = request.files.get('reg_json')
    if not json_f or not json_f.filename:
        return jsonify({'status': 'error', 'message': '请上传配准结果 JSON 文件'}), 400

    ext = os.path.splitext(json_f.filename)[1].lower()
    if ext not in ALLOWED_JSON:
        return jsonify({'status': 'error', 'message': '请上传 .json 文件'}), 400

    sd = get_sample_dir()
    if not sd:
        return jsonify({'status': 'error', 'message': '没有活跃样本'}), 400
    reg_dir = os.path.join(sd, 'reg')
    os.makedirs(reg_dir, exist_ok=True)

    json_path = os.path.join(reg_dir, 'rigid_result.json')
    json_f.save(json_path)

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        if 'affine_matrix' not in data:
            return jsonify({'status': 'error',
                            'message': 'JSON 中未找到 affine_matrix 字段'}), 400
        session['rigid_result_json'] = json_path
        M = np.array(data['affine_matrix'])
        return jsonify({
            'status' : 'success',
            'message': f'配准矩阵加载成功，缩放因子 ≈ {abs(M[0,0]):.2f}',
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ─── API：核心提取 ────────────────────────────────────────────────

@bp.route('/api/extract', methods=['POST'])
def extract():
    """
    根据前端提交的 ROI 多边形提取 MSI 像素数据。

    请求 JSON（新接口，推荐）
    -------------------------
    {
      "tissue_regions": [
        {
          "name"             : "<组织名>",
          "boundary_polygons": [[[x,y],...], ...] | null,
          "sub_rois": [
            {
              "name"    : "癌区1",
              "type"    : "cancer" | "paracancer" | "custom",
              "polygons": [[[x,y],...], ...]
            },
            ...
          ]
        },
        ...
      ],
      "pseudobulk_method": "median",
      "include_annotation": true,
      "use_tps": false
    }

    旧接口（兼容）：roi_list = [{name, type, polygon_img}, ...]
    """
    body = request.get_json(silent=True) or {}
    tissue_regions  = body.get('tissue_regions', None)
    roi_list_legacy = body.get('roi_list', None)
    pb_method = body.get('pseudobulk_method', 'mean')
    use_ann   = body.get('include_annotation', True)
    use_tps   = bool(body.get('use_tps', False))

    if not tissue_regions and not roi_list_legacy:
        return jsonify({'status': 'error', 'message': '请至少绘制一个 ROI 区域'}), 400

    # ── 1. 加载 MSI 数据 ──
    h5ad_path = get_h5ad_path()
    if not h5ad_path:
        return jsonify({'status': 'error',
                        'message': '请先在「数据导入」模块上传并归一化 MSI 数据'}), 400
    try:
        adata = ad.read_h5ad(h5ad_path)
    except Exception as e:
        return jsonify({'status': 'error',
                        'message': f'读取 MSI 数据失败: {e}'}), 500

    # ── 2. 加载配准矩阵 + 可选 TPS 控制点 ──
    rigid_path = _rigid_json_path()
    tps_src = tps_dst = None
    tps_applied = False
    if rigid_path and os.path.exists(rigid_path):
        try:
            affine_matrix = load_affine_matrix(rigid_path)
        except Exception as e:
            return jsonify({'status': 'error',
                            'message': f'读取配准矩阵失败: {e}'}), 500
        # 仅当前端请求 use_tps 且 JSON 里齐备时才启用 TPS
        if use_tps:
            try:
                with open(rigid_path, 'r', encoding='utf-8') as f:
                    rigid_data = json.load(f)
                src_pts = rigid_data.get('tps_src_points') or []
                dst_pts = rigid_data.get('tps_dst_points') or []
                if len(src_pts) >= 3 and len(src_pts) == len(dst_pts):
                    tps_src = src_pts
                    tps_dst = dst_pts
                    tps_applied = True
            except Exception:
                # 静默退化：TPS 字段读取失败时仍按 affine-only 走
                pass
    else:
        # 无配准矩阵时：使用单位矩阵（直接用 relative_x/y 作为图像坐标）
        affine_matrix = np.eye(3)

    # ── 3. 加载代谢物注释 ──
    annotation_df = _get_annotation_df() if use_ann else None

    # ── 4. ROI 提取 ──
    try:
        roi_results = extract_rois(
            adata, affine_matrix,
            tissue_regions=tissue_regions,
            roi_list=roi_list_legacy,
            tps_src_points=tps_src, tps_dst_points=tps_dst,
        )
    except Exception as e:
        logger.exception('roi extract error')
        return jsonify({'status': 'error', 'message': f'ROI 提取计算失败: {e}'}), 500

    if not roi_results:
        return jsonify({
            'status' : 'error',
            'message': '所有区域均未找到 MSI 像素，请检查多边形位置是否与配准图对应',
        }), 400

    # ── 5. 保存输出 CSV ──
    roi_dir = _ensure_roi_dir()
    if not roi_dir:
        return jsonify({'status': 'error', 'message': '没有活跃样本，无法保存结果'}), 400
    out_files = {}     # {roi_key: {intensity_csv, tissue_name, sub_roi_name, pixel_count}}
    preview_stats = []
    pb_rows = []       # 汇总 pseudo-bulk 行（所有区域放入一个文件）

    def _safe(s: str) -> str:
        return ''.join(c if c.isalnum() or c in '-_' else '_' for c in (s or ''))

    for roi_key, res in roi_results.items():
        tname = res.get('tissue_name', '') or ''
        sname = res.get('sub_roi_name', '') or roi_key
        # 文件名：组织名_子区域名（无组织名时省略）
        fn_stem = f'{_safe(tname)}__{_safe(sname)}' if tname else _safe(sname)

        # 像素级强度宽表
        intensity_df = build_intensity_dataframe(
            res['adata'],
            he_coords   = res['he_coords'],
            annotation_df = annotation_df,
        )
        int_csv = os.path.join(roi_dir, f'{fn_stem}_intensity.csv')
        intensity_df.to_csv(int_csv, index=False)

        # Pseudo-bulk 汇总行
        pb_df = compute_pseudobulk(res['adata'], method=pb_method,
                                   annotation_df=annotation_df)
        pb_df.insert(0, 'region_type', res['roi_type'])
        pb_df.insert(0, 'region_name', sname)
        pb_df.insert(0, 'tissue_name', tname)
        pb_df.insert(0, 'pixel_count', res['pixel_count'])
        pb_rows.append(pb_df)

        out_files[roi_key] = {
            'intensity_csv' : os.path.basename(int_csv),
            'pixel_count'   : res['pixel_count'],
            'tissue_name'   : tname,
            'sub_roi_name'  : sname,
        }
        preview_stats.append({
            'key'         : roi_key,
            'tissue_name' : tname,
            'sub_roi_name': sname,
            'type'        : res['roi_type'],
            'pixel_count' : res['pixel_count'],
            'n_mz'        : len(res['adata'].var_names),
        })

    # 合并 pseudo-bulk
    pb_all = pd.concat(pb_rows, ignore_index=True)
    pb_csv = os.path.join(roi_dir, 'pseudobulk_all.csv')
    pb_all.to_csv(pb_csv, index=False)

    # 顺手存一份本次提取使用的多边形定义（方便复现/重跑/审计）
    polygons_meta = {
        'tissue_regions'    : tissue_regions,
        'roi_list_legacy'   : roi_list_legacy,
        'use_tps_requested' : use_tps,
        'tps_applied'       : tps_applied,
        'pseudobulk_method' : pb_method,
    }
    try:
        with open(os.path.join(roi_dir, 'roi_polygons.json'), 'w', encoding='utf-8') as f:
            json.dump(polygons_meta, f, ensure_ascii=False, indent=2)
    except Exception:
        # 多边形快照写入失败不阻塞主流程
        logger.exception('failed to save roi_polygons.json')

    # 生成预览散点图（与提取走同一映射通路，避免空间漂移）
    try:
        preview_img = plot_roi_preview(
            adata, affine_matrix, roi_results,
            tps_src_points=tps_src, tps_dst_points=tps_dst,
        )
    except Exception:
        preview_img = None

    # 记录输出目录到 session（供下载路由使用）
    session['roi_result_dir'] = roi_dir
    mark_stage_done('roi')

    return jsonify({
        'status'      : 'success',
        'message'     : f'成功提取 {len(roi_results)} 个区域',
        'roi_stats'   : preview_stats,
        'out_files'   : out_files,
        'pseudobulk_csv': 'pseudobulk_all.csv',
        'preview_img' : preview_img,
        'roi_dir'     : roi_dir,
        'tps_applied' : tps_applied,
    })


# ─── API：下载 CSV ─────────────────────────────────────────────────

@bp.route('/api/download/<filename>')
def download(filename: str):
    """下载 ROI 提取结果 CSV"""
    # 优先用当前样本目录里的 roi/，回退 session 旧 key
    roi_dir = None
    sd = get_sample_dir()
    if sd:
        candidate = os.path.join(sd, 'roi')
        if os.path.isdir(candidate):
            roi_dir = candidate
    if not roi_dir:
        roi_dir = session.get('roi_result_dir')
    if not roi_dir:
        return jsonify({'status': 'error', 'message': '请先执行 ROI 提取'}), 404

    safe_fn = os.path.basename(filename).replace('\\', '')
    file_path = os.path.join(roi_dir, safe_fn)
    if not os.path.exists(file_path):
        return jsonify({'status': 'error', 'message': f'文件不存在: {safe_fn}'}), 404

    return send_file(
        file_path,
        as_attachment=True,
        download_name=safe_fn,
        mimetype='text/csv',
    )
