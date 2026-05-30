# MSI-Tool：胃癌空间代谢组学综合分析平台

## 项目定位

基于 Flask + 前端的本地 Web 应用，整合 AFADESI-MSI 数据处理与 HE 病理切片配准的全流程分析。最终打包为 exe，可离线演示。竞赛展示用途。

---

## 技术栈

- 后端：Python 3.10 + Flask
- 前端：HTML/CSS/JS（单页或多页均可，不用前端框架）
- 计算依赖：numpy, scipy, scikit-learn, opencv-python, matplotlib, pandas, openpyxl
- 打包：PyInstaller（最终阶段）

## 设计原则

- 每个功能模块独立成一个页面/tab，左侧导航栏切换
- 所有计算在后端 Python 完成，前端只做上传/展示/交互
- 结果图片由后端 matplotlib 生成，以 base64 或临时文件返回前端展示
- 用户数据存在本地临时目录，不需要数据库
- 中文界面

---

## 功能模块与开发顺序

### 第一阶段：纯 MSI 数据分析（不依赖配准）

#### 模块1：数据导入与预处理
- 输入：MSI 原始数据（txt 格式，像素×m/z 强度矩阵）+ 坐标文件
- 功能：
  - 读取并解析 MSI 数据为内部矩阵格式
  - TIC 归一化 / RMS 归一化（用户选择）
  - 显示归一化前后的 TIC 分布直方图
- 输出：归一化后的数据矩阵（内存中保持，供后续模块调用）

#### 模块2：降维与空间聚类
- 输入：模块1的归一化数据
- 功能：
  - PCA 降维（用户可选维度数）
  - KMeans / NMF 聚类（用户可选聚类数 k）
  - 聚类结果空间可视化（按像素坐标着色显示聚类标签）
  - UMAP 可视化（可选）
- 输出：聚类标签 + 空间聚类图

#### 模块3：空间积累模式分析（NMF）
- 输入：模块1的归一化数据
- 功能：
  - NMF 分解（用户可选 component 数）
  - 每个 component 的空间分布热图
  - 每个 component 的 top 代谢物权重列表
- 输出：NMF 空间模式图 + 权重表

### 第二阶段：MSI-HE 配准（已有基础代码）

#### 模块4：刚性配准
- 输入：MSI 空间数据 + HE 染色图像（PNG）
- 功能：
  - 自动轮廓提取（MSI 和 HE）
  - 外接圆计算 → 缩放因子
  - 质心对齐 → 平移
  - 旋转角度优化
  - 叠加预览（MSI 轮廓叠在 HE 上）
- 输出：配准参数（缩放、平移、旋转）+ 叠加图
- 备注：现有 generate_overlay_v4.py 可复用，需要封装为 Flask 路由

#### 模块5：TPS 精细配准
- 输入：模块4的粗配准结果
- 功能：
  - 手动标记控制点（前端交互：用户在 HE 和 MSI 上点击对应点）
  - 或自动控制点匹配
  - TPS 变换计算
  - 配准后叠加预览
- 输出：精细配准后的 MSI-HE 叠加图 + 变换矩阵

### 第三阶段：基于配准结果的联合分析

#### 模块6：ROI 标注与区域提取
- 输入：配准后的 MSI-HE 叠加图
- 功能：
  - 在 HE 图像上（或叠加图上）手动圈选 ROI（癌区 / 癌旁区）
  - 或导入 QuPath 等外部标注
  - 根据 ROI 提取对应 MSI 像素的代谢物强度数据
  - 按样本聚合（pseudo-bulk：取每个样本每个区域的中位数）
- 输出：癌 vs 癌旁的 pseudo-bulk 代谢物矩阵

#### 模块7：差异代谢物筛选
- 输入：模块6的 pseudo-bulk 矩阵
- 功能：
  - Pipeline_v2 三重筛选：
    - 配对 Wilcoxon 检验 → FDR 校正（Benjamini-Hochberg）
    - log₂FC 计算
    - PLS-DA → VIP 值
  - 筛选条件：FDR < 0.05 & |log₂FC| > 阈值 & VIP > 1（用户可调）
  - 火山图可视化
  - 差异代谢物列表导出（Excel）
- 输出：显著差异代谢物列表 + 火山图

#### 模块8：RF 组合分类器与 Biomarker 筛选
- 输入：模块7的差异代谢物 + pseudo-bulk 数据
- 功能：
  - RF 组合分类器 v2（嵌套 LOSO CV）
  - 样本级置换检验
  - 单代谢物 AUC 计算
  - Biomarker 决策表（Global AUC, Mean Sample-AUC, 置换 p 值, RF importance）
  - ROC 曲线可视化
- 输出：Biomarker 候选列表 + ROC 曲线 + 决策表

#### 模块9：差异代谢物空间热图
- 输入：配准结果 + 差异代谢物列表
- 功能：
  - 选择某个差异代谢物
  - 在 HE 图像上叠加该代谢物的空间强度热图
  - 癌区/癌旁区边界标注
  - 支持切换不同代谢物
- 输出：空间热图叠加图

---

## 项目文件结构

```
msi-tool/
├── app.py                  # Flask 主入口，注册所有蓝图
├── requirements.txt        # Python 依赖
├── config.py               # 全局配置（上传目录、临时目录等）
├── blueprints/
│   ├── data_import.py      # 模块1：数据导入与预处理
│   ├── clustering.py       # 模块2：降维与空间聚类
│   ├── nmf_patterns.py     # 模块3：空间积累模式
│   ├── rigid_reg.py        # 模块4：刚性配准
│   ├── tps_reg.py          # 模块5：TPS 精细配准
│   ├── roi_extract.py      # 模块6：ROI 标注与区域提取
│   ├── diff_metabolites.py # 模块7：差异代谢物筛选
│   ├── rf_classifier.py    # 模块8：RF 分类器
│   └── spatial_heatmap.py  # 模块9：空间热图
├── core/                   # 核心计算逻辑（纯 Python，不依赖 Flask）
│   ├── preprocessing.py
│   ├── clustering_algo.py
│   ├── nmf_analysis.py
│   ├── registration.py
│   ├── tps.py
│   ├── pseudobulk.py
│   ├── differential.py
│   ├── classifier.py
│   └── visualization.py
├── templates/
│   ├── base.html           # 基础模板（导航栏 + 布局）
│   ├── data_import.html
│   ├── clustering.html
│   ├── nmf.html
│   ├── rigid_reg.html
│   ├── tps_reg.html
│   ├── roi.html
│   ├── diff.html
│   ├── classifier.html
│   └── heatmap.html
├── static/
│   ├── css/
│   ├── js/
│   └── img/
└── temp/                   # 运行时临时文件（上传的数据、生成的图片）
```

## 代码规范

- Python 代码中文注释
- 每个 blueprint 文件顶部写清楚：该模块的输入、输出、依赖的其他模块
- core/ 下的计算逻辑不 import flask，保持纯粹，方便单独测试
- 前端页面风格统一：深色科研风格（延续现有 msi-tool 的视觉设计）
- 所有用户上传的文件存到 temp/ 目录，用 session 隔离不同用户

## 现有可复用代码

以下是部分已有的 Python 脚本，核心算法逻辑可以直接迁移到 core/ 目录：

- `ROC_AUC_组合分类器_final.py` → core/classifier.py（RF 分类器）
- `差异代谢物空间热图.py` → core/visualization.py（空间热图）

迁移时注意：把硬编码的文件路径改为函数参数传入，把 plt.show() 改为 plt.savefig() 返回图片。

## 开发优先级

1. 先搭好 app.py + base.html + 导航框架（能切换空页面）
2. 模块1（数据导入）→ 模块2（聚类）→ 模块3（NMF）：这三个不依赖配准，可以先做出来展示
3. 模块4（刚性配准）→ 模块5（TPS）：复用现有代码
4. 模块6-9：联合分析流程

## 安全设计（软件著作权申报相关）

工具需通过软件著作权审核，所有文件 I/O 类的接口均不接受用户指定的任意磁盘路径，统一采用"上传到 session 沙箱目录 + 服务端按固定文件名读取"的模式。

### 文件上传通用四层校验
所有接受用户文件的接口（如代谢物注释的数据库 CSV 上传）都实施以下安全策略：

1. **扩展名白名单** —— 从原始文件名取扩展名比对（避免中文文件名被 `secure_filename` 抹掉扩展名后绕过）
2. **大小双重校验** —— 先看 `Content-Length` 头快速失败，落盘后再用 `os.path.getsize` 权威复核（防伪造请求头）
3. **强制重命名** —— 不论用户上传文件叫什么，磁盘上一律落为预定义的固定名（杜绝路径注入和文件名特殊字符攻击）
4. **结构校验** —— 表格类文件用 `pd.read_csv(nrows=0)` 只读首行做列头校验（缺必需列直接拒收，且不会把伪装成 CSV 的大文件全读入内存）

### 沙箱隔离与生命周期
- 所有用户上传的文件落入 `temp/uploads/<sid>/`，不同 session 之间互不可见
- 启动时由 `app.py:_cleanup_old_sessions` 清理超过 `TEMP_MAX_AGE_DAYS = 7` 天未访问的 session 目录
- session 内文件采用"临时文件 + `os.replace` 原子替换"写入，避免半成品文件被读到

### 著作权说明书可引用的描述模板
> 数据库文件采用沙箱化上传机制，前端通过 `multipart/form-data` 提交文件，后端实施四层校验（扩展名白名单、大小限制、强制重命名、结构校验），文件落入按 session 隔离的临时目录，禁止用户指定任意文件系统路径，从根本上防止路径穿越和任意文件读取漏洞。

## 注意事项

- 工具最终用于软件著作权申报，文件 I/O 类接口需通过上一节的安全设计要求；其他方面（多用户并发、生产级部署）仍不需考虑
- 优先保证功能可用和演示效果，不追求性能优化
- 最终要用 PyInstaller 打包成 exe，所以避免使用不兼容 PyInstaller 的库
- matplotlib 图片生成时用 Agg 后端（非交互式），避免 GUI 弹窗