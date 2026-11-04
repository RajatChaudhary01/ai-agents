# Hosting-mode image for the AI agent. Built and run by deploy/docker-compose.yml.
FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

WORKDIR /srv/app

# Dependencies first so code changes don't invalidate the cached layer.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
COPY app.py LICENSE ./
RUN uv sync --locked --no-dev

# Unprivileged user; /data holds the accounts database (a mounted volume).
RUN useradd --system --uid 10001 --home-dir /nonexistent --shell /usr/sbin/nologin agent \
    && mkdir -p /data \
    && chown agent /data
USER agent

ENV PATH="/srv/app/.venv/bin:$PATH" \
    APP_MODE=host \
    APP_DB_PATH=/data/app.db \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_ANALYTICS_ENABLED=False \
    GRADIO_TEMP_DIR=/tmp/gradio \
    LITELLM_LOCAL_MODEL_COST_MAP=True \
    HF_HOME=/tmp/hf \
    XDG_CACHE_HOME=/tmp/cache \
    HOME=/tmp

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/healthz', timeout=4)"]

CMD ["python", "app.py", "--host"]
