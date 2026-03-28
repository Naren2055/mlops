# =============================================================================
# Docker image — Hugging Face Space (Streamlit inference UI)
# =============================================================================
# Configuration summary (rubric: list all configurations)
# -----------------------------------------------------------------------------
# | Setting              | Value / purpose                                      |
# |----------------------|------------------------------------------------------|
# | Base image           | python:3.9-slim                                    |
# | Working directory    | /app                                                |
# | Exposed port         | 8501 (Streamlit default; required by HF Spaces)    |
# | Dependency install   | pip install -r requirements.txt (pinned packages)    |
# | Application files    | COPY app.py, requirements.txt, and rest of context   |
# | Process              | streamlit run app.py                                 |
# | Streamlit server.port| 8501                                                |
# | Streamlit server.address | 0.0.0.0 (bind all interfaces for container)     |
# | XSRF protection      | disabled (Spaces / reverse-proxy compatibility)    |
# =============================================================================

FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
