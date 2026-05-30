"""
全局配置

数据根目录解析顺序（首个非空生效）：
  1. 环境变量 MSI_DATA_ROOT          —— 开发/部署时显式指定
  2. %LOCALAPPDATA%\msi-tool         —— Windows 标准用户数据位置
  3. <BASE_DIR>\temp                 —— 兜底，仅用于无 LOCALAPPDATA 的极少数情况

数据根目录下的子结构：
  DATA_ROOT/
  ├── library/
  │   └── upload_cache/   ← 按 SHA-256 缓存的原始文件，跨批次去重
  ├── batches/            ← 用户的批次（永久保留，不会自动清理）
  │   └── <batch_id>/
  │       ├── batch_meta.json
  │       ├── hmdb_db.csv (每批次共享一份)
  │       └── samples/<sample_id>/...
  └── tmp/                ← 真正的临时文件（截图、缩略图等），可被清理
"""
import os
import secrets

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_data_root() -> str:
    """按上述优先级解析数据根目录。结果保证存在。"""
    env_root = os.environ.get('MSI_DATA_ROOT')
    if env_root:
        return env_root
    local_appdata = os.environ.get('LOCALAPPDATA')
    if local_appdata:
        return os.path.join(local_appdata, 'msi-tool')
    return os.path.join(BASE_DIR, 'temp')


DATA_ROOT       = _resolve_data_root()
LIBRARY_FOLDER  = os.path.join(DATA_ROOT, 'library')
UPLOAD_CACHE    = os.path.join(LIBRARY_FOLDER, 'upload_cache')
BATCH_FOLDER    = os.path.join(DATA_ROOT, 'batches')
TMP_FOLDER      = os.path.join(DATA_ROOT, 'tmp')

# 兼容旧代码：保留 UPLOAD_FOLDER / RESULT_FOLDER 名字，指向 TMP_FOLDER
# 新代码应改用 BATCH_FOLDER + 样本目录。下面这些尾巴等阶段 B 全部模块迁移后再删除
UPLOAD_FOLDER  = TMP_FOLDER
RESULT_FOLDER  = TMP_FOLDER

# 上传相关
MAX_CONTENT_LENGTH = 500 * 1024 * 1024     # 单次请求最大 500MB
MAX_DB_SIZE        = 50 * 1024 * 1024      # 数据库 CSV 上限 50MB

# 批次软上限：超过会在 UI 给警告但不强制阻止
SAMPLE_SOFT_LIMIT = 50

# tmp 目录清理：启动时删除超过此天数的 tmp 子内容
# 注意：batches/ 永远不被自动清理，由用户在 UI 主动删
TMP_MAX_AGE_DAYS = 7

# Flask secret key（session 加密用）
# 优先读环境变量；否则持久化到 .secret_key 文件，保证重启后 session 不失效
_KEY_FILE = os.path.join(BASE_DIR, '.secret_key')
if os.environ.get('MSI_SECRET_KEY'):
    SECRET_KEY = os.environ['MSI_SECRET_KEY']
elif os.path.exists(_KEY_FILE):
    with open(_KEY_FILE) as _f:
        SECRET_KEY = _f.read().strip()
else:
    SECRET_KEY = secrets.token_hex(32)
    with open(_KEY_FILE, 'w') as _f:
        _f.write(SECRET_KEY)
    try:
        os.chmod(_KEY_FILE, 0o600)
    except OSError:
        pass

# 启动时确保所有持久目录存在
for _d in (DATA_ROOT, LIBRARY_FOLDER, UPLOAD_CACHE, BATCH_FOLDER, TMP_FOLDER):
    os.makedirs(_d, exist_ok=True)
