FROM python:3.9-slim

WORKDIR /app

# XGBoost wheels often need OpenMP at runtime on Debian slim (avoid silent crash on import).
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces: headless mode + no usage stats; reduces odd startup delays in some setups.
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# fileWatcherType=none saves CPU on read-only / ephemeral filesystems (common on Spaces).
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false", "--server.fileWatcherType", "none", "--browser.gatherUsageStats", "false"]
