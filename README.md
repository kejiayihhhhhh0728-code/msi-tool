---
title: PathoSpace-Met
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# PathoSpace-Met

病理组织空间代谢组学全流程分析平台（基于 Flask 的 Web 应用）。

整合 MSI 数据导入与预处理、代谢物注释、空间聚类、NMF 空间模式分析、
MSI–HE 配准、深度学习超分辨、ROI 提取、差异代谢物筛选、随机森林分类器、
空间热图与通路富集等模块。

- 源码仓库：<https://github.com/kejiayihhhhhh0728-code/msi-tool>
- 本地运行：`cd msi-tool && pip install -r requirements.txt && python app.py` → <http://localhost:5000>
- 在线部署：本仓库根目录的 `Dockerfile` 用于 Hugging Face Spaces（Docker SDK，端口 7860）

## 在线演示说明

- 免费 Space 无持久存储，容器重启后上传的数据与会话会清空，仅供功能演示。
- 「超分辨」模块依赖训练好的 VDSR 权重文件（未随仓库发布），在线站该功能不可用，
  其余模块可正常体验。

## 核心功能

- MSI 原始数据导入、TIC/RMS 归一化与质量预览
- 代谢物注释、降维聚类与空间 NMF 模式分析
- MSI 与 HE 病理图像的刚性配准和 TPS 精配准
- 纯 VDSR、VDSR + HE 监督、VDSR + HE + LCRN 三种超分辨模型应用
- 基于配准结果的 ROI 标注、pseudo-bulk 提取与差异代谢物分析
- 随机森林 biomarker 筛选、空间热图叠加与通路富集分析
