---
title: Wellness Tourism Purchase Predictor
emoji: 🧳
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
startup_duration_timeout: 45m
---

Front-end Streamlit app for tourism prediction. The UI lives in **`src/streamlit_app.py`**; the `Dockerfile` uses **`ENTRYPOINT`** with **`HEALTHCHECK`** on `/_stcore/health` (Python **3.13.5-slim** base).

### Build looks stuck / no logs?

On **Docker** Spaces, the Hub often **does not stream** output inside a `RUN` until that **entire layer** finishes. Installing `scikit-learn`, `xgboost`, and `streamlit` can take **many minutes** on the builder with a **blank log** — that is normal.

- Wait at least **15–20 minutes** on the first build; `startup_duration_timeout` above is **45m**.
- Refresh the **Build logs** tab; try another browser or incognito if the page hangs.
- After a failed build, use **Factory reboot** (Space settings) and rebuild.
- Check [Hugging Face status](https://status.huggingface.co/) for incidents or queue delays.

If builds are repeatedly slow, open **Space → Settings → Dev mode** (if available) or build the same `Dockerfile` locally with `docker build` to see full `pip` output.