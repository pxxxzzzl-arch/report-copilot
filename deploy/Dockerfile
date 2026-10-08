# 应用镜像：api 与 ui 共用同一个镜像，由 docker-compose.yml 中的 command 决定启动哪个服务（D-031）。
# 模型权重、年报 PDF、切分结果都不打进镜像，运行时以只读卷挂载到 /app/data。
#
# 一般不需要单独构建：deploy/scripts/up.sh 会通过 docker compose build 按本机架构构建。单独构建：
#   docker build -f deploy/Dockerfile -t report-copilot:local .

# ---------- 构建阶段：用 uv 按锁文件安装依赖 ----------
FROM python:3.11.16-slim-bookworm AS builder

# 与本地开发一致的 uv 版本（从 PyPI 安装，不依赖 ghcr.io）
RUN pip install --no-cache-dir uv==0.12.18

WORKDIR /app
# UV_COMPILE_BYTECODE：安装时预编译 .pyc，缩短容器启动时间；UV_LINK_MODE=copy：缓存挂载与目标目录不在同一文件系统
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/app/.venv

# 先只复制依赖声明：代码变动时这一层可以命中缓存，不用重新安装依赖
COPY pyproject.toml uv.lock ./
# --frozen：严格按 uv.lock 安装，不重新解析；--no-dev：不装 pytest 等开发依赖
# Linux 上 torch 来自 PyTorch 的 CPU 源（pyproject.toml 的 tool.uv.sources），不会装 CUDA
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# ---------- 运行阶段 ----------
FROM python:3.11.16-slim-bookworm

# libgomp1：PyTorch CPU 版的 OpenMP 运行库（slim 镜像没有）
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 以非 root 用户运行：即使应用被攻破，也不能直接获得容器内的 root 权限
RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --chown=app:app . .

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # 模型只从挂载的本地目录加载，禁止运行时联网下载
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/tmp/hf

USER app
EXPOSE 8000 8501

# 默认启动 API；ui 服务在 compose 中覆盖 command
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
