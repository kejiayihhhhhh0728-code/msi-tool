# -*- coding: utf-8 -*-
"""
实验记录 20 - 表 6-1 定量评估结果可视化（2×2 + 顶部图例版本）
方法名移到顶部统一图例，x 轴留空。适合单栏宽度排版。
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.patches import Rectangle

# =============================================================
# 1. 全局绘图参数
# =============================================================
rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 9
rcParams['axes.linewidth'] = 0.8
rcParams['xtick.major.width'] = 0.8
rcParams['ytick.major.width'] = 0.8
rcParams['xtick.major.size'] = 3
rcParams['ytick.major.size'] = 3
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['savefig.bbox'] = 'tight'
rcParams['savefig.pad_inches'] = 0.05

# =============================================================
# 2. 数据（来源：实验记录 20 表 6-1）
# =============================================================
methods_full = ['Bicubic',
                'Vanilla VDSR',
                'VDSR+HE (w/o LCRN)',
                'Proposed (VDSR+LCRN)']

psnr = [17.2809, 19.1620, 19.2685, 19.5998]   # ↑
ssim = [0.6993,  0.7220,  0.7245,  0.7269]    # ↑
sam  = [0.2429,  0.1780,  0.1755,  0.1642]    # ↓
rmse = [0.1609,  0.1437,  0.1419,  0.1387]    # ↓

colors = ['#B8B8B8', '#9EC5E8', '#4A89C7', '#E07B39']

# =============================================================
# 3. 绘图（2×2 布局）
# =============================================================
fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.4), dpi=300)
axes = axes.flatten()

metrics = [
    {'title': r'PSNR (dB) $\uparrow$',   'data': psnr, 'fmt': '{:.4f}', 'better': 'high'},
    {'title': r'SSIM $\uparrow$',        'data': ssim, 'fmt': '{:.4f}', 'better': 'high'},
    {'title': r'SAM (rad) $\downarrow$', 'data': sam,  'fmt': '{:.4f}', 'better': 'low'},
    {'title': r'RMSE $\downarrow$',      'data': rmse, 'fmt': '{:.4f}', 'better': 'low'},
]

x_pos = np.arange(len(methods_full))
bar_width = 0.65

for ax, m in zip(axes, metrics):
    bars = ax.bar(x_pos, m['data'],
                  width=bar_width,
                  color=colors,
                  edgecolor='black',
                  linewidth=0.7,
                  zorder=3)

    data_range = max(m['data']) - min(m['data'])
    if m['better'] == 'high':
        ymin = min(m['data']) - data_range * 0.20
        ymax = max(m['data']) + data_range * 0.30
    else:
        ymin = min(m['data']) - data_range * 0.25
        ymax = max(m['data']) + data_range * 0.25
    ax.set_ylim(ymin, ymax)

    # 在柱顶标注数值
    for bar, val in zip(bars, m['data']):
        ax.text(bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + data_range * 0.03,
                m['fmt'].format(val),
                ha='center', va='bottom',
                fontsize=7.5, color='black')

    ax.set_title(m['title'], fontsize=10.5, pad=6, fontweight='bold')

    # x 轴留空，方法名由顶部 legend 统一表达
    ax.set_xticks(x_pos)
    ax.set_xticklabels([])
    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', labelsize=8)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', linewidth=0.4, color='gray', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

# =============================================================
# 4. 顶部统一图例
# =============================================================
legend_handles = [Rectangle((0, 0), 1, 1,
                            facecolor=c, edgecolor='black', linewidth=0.7)
                  for c in colors]

fig.legend(legend_handles, methods_full,
           loc='upper center',
           ncol=4,
           bbox_to_anchor=(0.5, 1.00),
           frameon=False,
           fontsize=9,
           handlelength=1.4,
           handleheight=1.0,
           columnspacing=1.8)

# rect 顶部留 7% 给 legend
plt.tight_layout(rect=[0, 0, 1, 0.93], w_pad=2.2, h_pad=2.5)

plt.savefig('figure_table6_1_comparison_2x2.pdf', dpi=300)
plt.savefig('figure_table6_1_comparison_2x2.png', dpi=300)
plt.show()
print('Figure saved: figure_table6_1_comparison_2x2.pdf / .png')