---
title: Wellness Tourism Purchase Predictor
emoji: 🧳
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.43.2
---

Use this file **only** if your Hugging Face Space uses the **Streamlit** SDK (not Docker).

**Steps**

1. In the Space repo on Hugging Face, **delete** `Dockerfile` (Streamlit Spaces use HF’s managed runtime).
2. Rename this file to **`README.md`** (replace the Docker README), or paste the YAML block above into `README.md`.
3. Keep **`app.py`** and **`requirements.txt`** in the Space root.
4. Set Space **Repository secrets**: `HF_MODEL_REPO`, and `HF_TOKEN` if the model repo is private.

Port **8501** is enforced by Hugging Face for Streamlit Spaces; do not override it in a `.streamlit/config.toml`.
