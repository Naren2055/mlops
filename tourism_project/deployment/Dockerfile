# Hugging Face Space (Docker SDK): listen on 0.0.0.0:7860 (must match README app_port).
# Base Python 3.10 matches GitHub Actions / training (tourism_project/requirements.txt).
# - curl: HEALTHCHECK
# - libgomp1: XGBoost shared library on Debian slim
# Build logs often stay empty until each RUN layer completes (pip can take many minutes).
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Verbose logs in `docker logs` / Space build & runtime logs.
ENV STREAMLIT_LOGGER_LEVEL=debug \
    HF_HUB_VERBOSITY=debug \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PYTHONUNBUFFERED=1

COPY requirements.txt ./
COPY src/ ./src/

RUN echo "=== HF build: pip install start ===" \
    && pip3 install --no-cache-dir --upgrade pip \
    && pip3 install --no-cache-dir --prefer-binary -r requirements.txt \
    && echo "=== HF build: pip install finished ==="

COPY . .

EXPOSE 7860

# Requires curl (installed above). start-period allows Streamlit to boot before first check.
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD curl --fail --silent http://127.0.0.1:7860/_stcore/health || exit 1

# --server.address=0.0.0.0 is required: otherwise Streamlit binds to localhost only and the Space proxy cannot reach the app.
ENTRYPOINT ["streamlit", "run", "src/streamlit_app.py", "--logger.level=debug", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false", "--server.fileWatcherType", "none", "--browser.gatherUsageStats", "false"]
