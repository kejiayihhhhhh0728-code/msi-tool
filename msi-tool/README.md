# PathoSpace-Met

病理组织空间代谢组学全流程分析平台。

PathoSpace-Met 是一个基于 Flask 的本地 Web 应用，面向病理组织空间代谢组学研究流程，整合 MSI 数据导入与预处理、代谢物注释、空间聚类、NMF 空间模式分析、MSI-HE 配准、ROI 提取、差异代谢物筛选、随机森林分类器、空间热图和通路富集等模块。

## 快速启动

```bash
pip install -r requirements.txt
python app.py
# 浏览器访问 http://localhost:5000
```

## 核心功能

- MSI 原始数据导入、TIC/RMS 归一化与质量预览
- 代谢物注释、降维聚类与空间 NMF 模式分析
- MSI 与 HE 病理图像的刚性配准和 TPS 精配准
- 基于配准结果的 ROI 标注、pseudo-bulk 提取与差异代谢物分析
- 随机森林 biomarker 筛选、空间热图叠加与通路富集分析

## 文件结构

```text
msi-tool/
├── app.py                  # Flask 主入口
├── requirements.txt        # Python 依赖
├── config.py               # 全局配置
├── blueprints/             # 页面路由与 API
├── core/                   # 核心计算逻辑
├── templates/              # 前端模板
├── static/                 # 样式与静态资源
├── data/                   # 内置数据库与模型文件
└── temp/                   # 运行时临时文件
```
