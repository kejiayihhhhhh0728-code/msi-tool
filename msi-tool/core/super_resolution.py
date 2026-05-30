from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

IMAGE_EXTS = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')

MODEL_VARIANTS = ('vanilla', 'he_guided', 'lcrn_guided')
MODEL_LABELS = {
    'vanilla': '纯 VDSR',
    'he_guided': 'VDSR + HE 监督',
    'lcrn_guided': 'VDSR + HE + LCRN',
}

DEFAULT_MODEL_PATHS = {
    'vanilla': r'C:\Users\Lenovo\Desktop\Results\VDSR-naked-noneHE\800epoch\vdsr_model_vanilla2.pth',
    'he_guided': r'C:\Users\Lenovo\Desktop\Results\VDSR-naked-HE\800epoch\vdsr_model_he_only2.pth',
    'lcrn_guided': r'C:\Users\Lenovo\Desktop\Results\VDSR-LCRN-GE\800epoch\vdsr_model_LCRN-ge-final-3_guided.pth',
}


@dataclass
class SuperResolutionConfig:
    lr_msi_path: str
    output_dir: str
    model_variant: str = 'lcrn_guided'
    he_path: str = ''
    upscale_factor: int = 1
    model_path: str = ''


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _safe_token(value: str) -> str:
    out = []
    for ch in str(value):
        out.append(ch if ch.isalnum() or ch in ('-', '_', '.') else '_')
    return ''.join(out).strip('_') or 'mz'


def _normalize_channel(values: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
    """返回 [0,1] 范围的 float32 强度，便于后续应用 colormap。"""
    values = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    finite = values[np.isfinite(values)]
    positive = finite[finite > 0]
    ref = positive if positive.size else finite
    if ref.size == 0:
        return np.zeros_like(values, dtype=np.float32)

    lo = float(np.percentile(ref, low_pct))
    hi = float(np.percentile(ref, high_pct))
    if hi <= lo:
        hi = float(ref.max())
        lo = float(ref.min())
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)

    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _msi_colormap():
    """与 core/preprocessing._msi_cmap 同款：深蓝 → 蓝 → 青 → 绿 → 黄 → 橙 → 红。
    与训练数据的 MSI 热图着色保持一致，避免推理时分布偏移导致输出颜色异常。"""
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#00008B', '#0000FF', '#00FFFF', '#00FF00',
              '#FFFF00', '#FFA500', '#FF0000']
    return LinearSegmentedColormap.from_list('msi_cmap', colors)


def _nearest_mz_index(adata, target_mz: float) -> tuple[int, float]:
    if 'm/z' in adata.var.columns:
        mz_values = adata.var['m/z'].astype(str).str.strip().astype(float).to_numpy()
    else:
        mz_values = np.asarray([float(str(v).strip()) for v in adata.var_names], dtype=np.float64)
    idx = int(np.nanargmin(np.abs(mz_values - float(target_mz))))
    return idx, float(mz_values[idx])


def render_mz_channel_image(
    h5ad_path: str,
    target_mz: float,
    output_dir: str,
    tolerance: float | None = None,
) -> Dict:
    """
    Render one m/z channel from the current sample AnnData as an RGB image for SR inference.
    """
    import anndata as ad

    if not h5ad_path or not os.path.exists(h5ad_path):
        raise FileNotFoundError('未找到当前样本的 MSI h5ad 数据，请先完成数据导入/归一化')

    adata = ad.read_h5ad(h5ad_path)
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError('h5ad 数据为空，无法生成 m/z 图像')
    if 'relative_x' not in adata.obs or 'relative_y' not in adata.obs:
        raise ValueError("h5ad 缺少 relative_x / relative_y 空间坐标")

    idx, actual_mz = _nearest_mz_index(adata, target_mz)
    delta = abs(actual_mz - float(target_mz))
    if tolerance is not None and tolerance > 0 and delta > tolerance:
        raise ValueError(f'未找到容差内的 m/z：目标 {target_mz}, 最近 {actual_mz:.6f}, 差值 {delta:.6f}')

    x_raw = np.asarray(adata.obs['relative_x'], dtype=np.float64)
    y_raw = np.asarray(adata.obs['relative_y'], dtype=np.float64)
    x_vals, x = np.unique(x_raw, return_inverse=True)
    y_vals, y = np.unique(y_raw, return_inverse=True)
    width = int(len(x_vals))
    height = int(len(y_vals))
    if width <= 0 or height <= 0:
        raise ValueError('空间坐标范围异常，无法生成图像')

    col = adata.X[:, idx]
    if hasattr(col, 'toarray'):
        col = col.toarray()
    values = np.asarray(col).reshape(-1)
    intensities = _normalize_channel(values)  # float32 ∈ [0,1]
    canvas = np.zeros((height, width), dtype=np.float32)
    mask = np.zeros((height, width), dtype=bool)
    canvas[y, x] = intensities
    mask[y, x] = True

    # 应用 MSI 色图把强度图渲染成彩色 RGB（与训练分布一致）
    cmap = _msi_colormap()
    rgba = cmap(canvas)  # H x W x 4, 取值 [0,1]
    rgb = (rgba[..., :3] * 255.0).round().astype(np.uint8)
    # 不在样本范围内的像素保留黑色背景，避免被 colormap 染成深蓝
    rgb[~mask] = 0

    input_dir = _ensure_dir(output_dir)
    stem = f'mz_{_safe_token(f"{actual_mz:.6f}")}_from_h5ad'
    image_path = os.path.join(input_dir, f'{stem}.png')
    Image.fromarray(rgb, mode='RGB').save(image_path)

    return {
        'image_path': image_path,
        'source_h5ad': h5ad_path,
        'target_mz': float(target_mz),
        'actual_mz': actual_mz,
        'mz_delta': delta,
        'var_index': idx,
        'image_size': [width, height],
        'nonzero_pixels': int(np.count_nonzero(values)),
    }


def get_model_status() -> Dict[str, Dict]:
    return {
        key: {
            'label': MODEL_LABELS[key],
            'path': path,
            'exists': os.path.exists(path),
            'requires_he': key != 'vanilla',
        }
        for key, path in DEFAULT_MODEL_PATHS.items()
    }


def _build_model(variant: str):
    import torch.nn as nn
    import torch.nn.functional as F

    class SFTLayer(nn.Module):
        def __init__(self, msi_channels=64, he_channels=64):
            super().__init__()
            self.cond_conv1 = nn.Conv2d(he_channels, 32, 3, 1, 1)
            self.cond_conv2 = nn.Conv2d(32, 32, 3, 1, 1)
            self.scale_conv = nn.Conv2d(32, msi_channels, 3, 1, 1)
            self.shift_conv = nn.Conv2d(32, msi_channels, 3, 1, 1)

        def forward(self, msi_feat, he_cond):
            c = F.leaky_relu(self.cond_conv1(he_cond), 0.1)
            c = F.leaky_relu(self.cond_conv2(c), 0.1)
            scale = self.scale_conv(c)
            shift = self.shift_conv(c)
            return msi_feat * (scale + 1) + shift

    class LCRN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.LeakyReLU(0.1, inplace=True),
            )

        def forward(self, he_img):
            return self.net(he_img)

    class VanillaVDSR(nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU(inplace=True)
            self.conv_in = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
            self.mid_convs = nn.ModuleList([
                nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False) for _ in range(18)
            ])
            self.conv_out = nn.Conv2d(64, 3, kernel_size=3, padding=1, bias=False)

        def forward(self, lr_msi):
            x = self.relu(self.conv_in(lr_msi))
            for conv in self.mid_convs:
                x = self.relu(conv(x))
            residual = self.conv_out(x)
            return lr_msi + residual

    class HENoLayerVDSR(nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU(inplace=True)
            self.conv_in = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
            self.mid_convs = nn.ModuleList([
                nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False) for _ in range(18)
            ])
            self.sft1 = SFTLayer(msi_channels=64, he_channels=3)
            self.sft2 = SFTLayer(msi_channels=64, he_channels=3)
            self.sft3 = SFTLayer(msi_channels=64, he_channels=3)
            self.conv_out = nn.Conv2d(64, 3, kernel_size=3, padding=1, bias=False)

        def forward(self, lr_msi, he_img):
            x = self.relu(self.conv_in(lr_msi))
            for i, conv in enumerate(self.mid_convs):
                x = self.relu(conv(x))
                if i == 5:
                    x = self.sft1(x, he_img)
                elif i == 11:
                    x = self.sft2(x, he_img)
                elif i == 17:
                    x = self.sft3(x, he_img)
            residual = self.conv_out(x)
            return lr_msi + residual

    class GuidedVDSR(nn.Module):
        def __init__(self):
            super().__init__()
            self.lcrn = LCRN()
            self.relu = nn.ReLU(inplace=True)
            self.conv_in = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
            self.mid_convs = nn.ModuleList([
                nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False) for _ in range(18)
            ])
            self.sft1 = SFTLayer(msi_channels=64, he_channels=64)
            self.sft2 = SFTLayer(msi_channels=64, he_channels=64)
            self.sft3 = SFTLayer(msi_channels=64, he_channels=64)
            self.conv_out = nn.Conv2d(64, 3, kernel_size=3, padding=1, bias=False)

        def forward(self, lr_msi, he_img):
            he_cond = self.lcrn(he_img)
            x = self.relu(self.conv_in(lr_msi))
            for i, conv in enumerate(self.mid_convs):
                x = self.relu(conv(x))
                if i == 5:
                    x = self.sft1(x, he_cond)
                elif i == 11:
                    x = self.sft2(x, he_cond)
                elif i == 17:
                    x = self.sft3(x, he_cond)
            residual = self.conv_out(x)
            return lr_msi + residual

    if variant == 'vanilla':
        return VanillaVDSR()
    if variant == 'he_guided':
        return HENoLayerVDSR()
    if variant == 'lcrn_guided':
        return GuidedVDSR()
    raise ValueError(f'未知模型类型：{variant}')


def _to_tensor(img: Image.Image):
    import torch

    arr = np.asarray(img.convert('RGB'), dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)


def _tensor_to_image(tensor) -> Image.Image:
    arr = tensor.squeeze(0).detach().cpu().clamp(0, 1).numpy()
    arr = (arr.transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode='RGB')


def apply_super_resolution(config: SuperResolutionConfig) -> Dict:
    import torch

    if config.model_variant not in MODEL_VARIANTS:
        raise ValueError(f'模型类型必须是 {", ".join(MODEL_VARIANTS)}')
    if not config.lr_msi_path or not os.path.exists(config.lr_msi_path):
        raise ValueError('请上传 LR MSI 图像')

    model_path = config.model_path or DEFAULT_MODEL_PATHS[config.model_variant]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'未找到模型权重：{model_path}')
    if config.model_variant != 'vanilla' and (not config.he_path or not os.path.exists(config.he_path)):
        raise ValueError('HE 监督模型需要 HE 图像，可上传 HE 或先完成刚性配准')

    out_dir = _ensure_dir(config.output_dir)
    pred_dir = _ensure_dir(os.path.join(out_dir, 'predictions'))
    started_at = datetime.now().isoformat(timespec='seconds')

    lr_img = Image.open(config.lr_msi_path).convert('RGB')
    input_size = lr_img.size
    if int(config.upscale_factor) > 1:
        factor = int(config.upscale_factor)
        lr_img = lr_img.resize((lr_img.width * factor, lr_img.height * factor), Image.Resampling.BICUBIC)

    he_img = None
    he_resized = False
    if config.model_variant != 'vanilla':
        he_img = Image.open(config.he_path).convert('RGB')
        if he_img.size != lr_img.size:
            he_img = he_img.resize(lr_img.size, Image.Resampling.BICUBIC)
            he_resized = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = _build_model(config.model_variant).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    lr = _to_tensor(lr_img).to(device)
    he = _to_tensor(he_img).to(device) if he_img else None
    with torch.inference_mode():
        sr = model(lr, he) if config.model_variant != 'vanilla' else model(lr)

    stem = os.path.splitext(os.path.basename(config.lr_msi_path))[0]
    output_tiff = os.path.join(pred_dir, f'{stem}_{config.model_variant}_sr.tiff')
    preview_png = os.path.join(pred_dir, f'{stem}_{config.model_variant}_preview.png')
    lr_preview_png = os.path.join(pred_dir, f'{stem}_{config.model_variant}_lr_preview.png')
    sr_img = _tensor_to_image(sr)
    sr_img.save(output_tiff)
    sr_img.save(preview_png)
    # 保存模型实际输入的图像（已按 upscale_factor 做 bicubic 预放大），
    # 与 SR 输出尺寸一致，便于前端并排对比
    lr_img.save(lr_preview_png)

    result = {
        'started_at': started_at,
        'finished_at': datetime.now().isoformat(timespec='seconds'),
        'config': asdict(config),
        'variant': config.model_variant,
        'variant_label': MODEL_LABELS[config.model_variant],
        'device': str(device),
        'model_path': model_path,
        'input_path': config.lr_msi_path,
        'he_path': config.he_path,
        'input_size': input_size,
        'output_size': sr_img.size,
        'lr_preview_size': lr_img.size,
        'upscale_factor': int(config.upscale_factor),
        'he_resized': he_resized,
        'output_path': output_tiff,
        'preview_path': preview_png,
        'lr_preview_path': lr_preview_png,
    }
    summary_path = os.path.join(out_dir, 'super_resolution_result.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    result['summary_path'] = summary_path
    return result
