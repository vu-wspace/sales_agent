# Adapter Zalo OA (cổng 8003). Build context: thư mục gốc repo
#   docker build -f services/zalo/Dockerfile .
FROM python:3.12.15-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    ZALO_DB_PATH=/data/zalo.sqlite3

WORKDIR /app

# CA nội bộ cho mạng có SSL inspection (xem services/agent/Dockerfile)
COPY certs/ /usr/local/share/ca-certificates/
RUN update-ca-certificates
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

COPY services/zalo/pyproject.toml services/zalo/uv.lock services/zalo/
RUN --mount=from=ghcr.io/astral-sh/uv:0.12.23,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --project services/zalo --no-install-project

COPY services/zalo ./services/zalo
RUN --mount=from=ghcr.io/astral-sh/uv:0.12.23,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --project services/zalo

# /data: SQLite của adapter (người dùng Zalo -> thread_id, token Zalo). Gắn volume để không mất khi tạo lại container
RUN useradd --create-home --uid 1000 app && mkdir /data && chown app /data
USER app

EXPOSE 8003
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8003/health', timeout=4)"

# 1 worker: SQLite + khóa theo từng khách nằm trong process
CMD ["uvicorn", "phone_zalo.app:app", "--host", "0.0.0.0", "--port", "8003", "--workers", "1"]
