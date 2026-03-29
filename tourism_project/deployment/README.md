---
title: "Wellness Tourism Purchase Predictor"
emoji: 🧳
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
startup_duration_timeout: 45m
---

Front-end Streamlit app for tourism prediction. The UI lives in **`src/streamlit_app.py`**; the `Dockerfile` uses **`ENTRYPOINT`** with **`HEALTHCHECK`** on `/_stcore/health` (Python **3.13.5-slim** base).
