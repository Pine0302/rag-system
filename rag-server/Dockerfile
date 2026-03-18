FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖（避免构建时缺失）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 升级 pip（解决依赖解析问题）
RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .

# 先安装轻量依赖，最后装 torch（最大）
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
