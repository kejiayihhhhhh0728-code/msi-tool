# PathoSpace-Met —— Hugging Face Spaces (Docker SDK) 部署镜像
FROM python:3.10-slim

# 系统依赖：opencv-python 运行时需要 libGL 与 glib
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 安装 Python 依赖 + 生产级 WSGI 服务器
COPY msi-tool/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# 拷贝应用代码（内层 msi-tool/ 即 Flask 应用根目录）
COPY msi-tool/ ./

# 数据目录指向容器内可写的临时位置（HF 免费 Space 无持久盘，重启即清空）
# secret key 用环境变量注入，避免向只读位置写 .secret_key
ENV MSI_DATA_ROOT=/tmp/msi-data \
    PYTHONUNBUFFERED=1

# HF Spaces 通过 app_port 暴露此端口
EXPOSE 7860

# 用 gunicorn 绑定 0.0.0.0:7860（不改动 app.py 里的 app.run）
# torch 导入较慢，给足启动 timeout；单 worker 控内存，多线程应付并发
CMD ["gunicorn", "app:app", \
     "-b", "0.0.0.0:7860", \
     "-k", "gthread", \
     "--workers", "1", \
     "--threads", "4", \
     "--timeout", "600"]
