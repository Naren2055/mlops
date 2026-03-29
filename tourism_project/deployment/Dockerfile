# Python 3.13 + Streamlit on port 8501 (matches Space README app_port).
# HF may buffer logs until each RUN layer completes; pip can take several minutes.
FROM python:3.13.5-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PYTHONUNBUFFERED=1

COPY requirements.txt ./
COPY src/ ./src/

RUN echo "=== HF build: pip install start ===" \
    && pip3 install --no-cache-dir --prefer-binary -r requirements.txt \
    && echo "=== HF build: pip install finished ==="

# README, optional hf_http_config from upload, Dockerfile (harmless in image).
COPY . .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl --fail --silent http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false", "--server.fileWatcherType", "none", "--browser.gatherUsageStats", "false"]
