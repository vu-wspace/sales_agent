# Backend cửa hàng (đơn hàng, phiên, email). Không có torch/embedding. Build context: thư mục gốc repo
#   docker build -f services/be/Dockerfile .
FROM python:3.12.15-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# CA nội bộ cho mạng có SSL inspection (xem services/agent/Dockerfile)
COPY certs/ /usr/local/share/ca-certificates/
RUN update-ca-certificates
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

COPY pyproject.toml uv.lock ./
COPY packages/common/pyproject.toml packages/common/
COPY services/agent/pyproject.toml services/agent/
COPY services/be/pyproject.toml services/be/
RUN --mount=from=ghcr.io/astral-sh/uv:0.12.23,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --package phone-api --no-install-workspace

COPY packages/common ./packages/common
COPY services/be ./services/be
COPY data_schema/orders.sql ./data_schema/orders.sql
RUN --mount=from=ghcr.io/astral-sh/uv:0.12.23,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --package phone-api

RUN useradd --create-home --uid 1000 app
USER app

EXPOSE 8002
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8002/health', timeout=4)"

# be không giữ trạng thái trong RAM (giỏ, lịch sử ở agent; đơn ở Postgres) nên tăng worker/replica được;
# để 1 vì orders.ensure_tables (mã hóa dữ liệu cũ) lúc khởi động không nên chạy song song
CMD ["uvicorn", "phone_api.api:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "1"]
