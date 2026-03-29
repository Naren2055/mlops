# Wellness Tourism MLOps (Visit with Us)

End-to-end pipeline: register data on Hugging Face, prepare splits, train/tune **XGBoost** with **MLflow** tracking, push the model to the Hub, and deploy a **Streamlit** app via **Docker** on a Hugging Face Space. **GitHub Actions** (`.github/workflows/pipeline.yml`) runs the same steps on every push to **`main`**.

## Repository layout (rubric: master folder + `data`)

```text
.github/workflows/pipeline.yml   # CI/CD: register → prep → train → deploy
tourism.csv                      # Raw data (optional at root; copied into data/)
tourism_project/
  data/                          # Subfolder "data": tourism.csv, train/test CSVs
  model_building/
    data_register.py             # Upload data/ → HF Dataset
    prep.py                      # Load from HF, clean, split, save + upload splits
    train.py                     # GridSearch + MLflow + model upload
  deployment/
    Dockerfile
    README.md                    # YAML: sdk docker, app_port 7860 (HF-friendly port)
    requirements.txt
    src/
      streamlit_app.py           # Streamlit; Hub download + DataFrame + predict_proba
  hosting/hosting.py             # Upload deployment/ → HF Space
  requirements.txt               # Python deps for Actions & notebook scripts
tourism_prediction_mlops_narendrababu_S.ipynb
```

## Hugging Face repositories (defaults)

| Type    | Repo id (replace `USER`)              |
|---------|----------------------------------------|
| Dataset | `USER/wellness-tourism-purchase`       |
| Model   | `USER/wellness-tourism-xgboost-model`  |
| Space   | `USER/wellness-tourism-streamlit`      |

## GitHub Actions secrets

- `HF_TOKEN` — Hugging Face token (write)
- `HF_USER` — Hugging Face username (string)

## Space configuration (Docker only)

- **SDK:** Space must be **Docker** (matches course template: Docker + Streamlit).
- **Python:** The Space image uses **Python 3.10** (`python:3.10-slim`), matching **GitHub Actions** and reducing pickle / sklearn drift versus the trained `joblib` artifact.
- **Port:** `deployment/Dockerfile` and `deployment/README.md` both use **`7860`**. `app_port` in `README.md` must match `--server.port` in the `Dockerfile` or the Space can stay stuck on **Starting**.
- **Layout:** After `hosting.py`, the Space repo root must contain **`README.md`**, **`Dockerfile`**, **`requirements.txt`**, **`src/streamlit_app.py`**, and optionally **`hf_http_config.py`**.
- **Secrets:** Space **Repository secrets** — `HF_MODEL_REPO` (and `HF_TOKEN` if the model repo is private).
- **Verify deploy:** Run `python tourism_project/hosting/hosting.py` after a green CI run (or from the notebook). On the Space, **README** must show `sdk: docker` and **`app_port: 7860`**, and the repo must contain **`Dockerfile`**, **`requirements.txt`**, **`src/streamlit_app.py`**, and **`hf_http_config.py`** (staged automatically by `hosting.py`).

## Local / notebook

Set `HF_TOKEN` and `HF_USER`, then run the notebook **sequentially** (or run the Python scripts under `tourism_project/model_building/` and `tourism_project/hosting/`).

If Hub calls fail with `SSLCertVerificationError` (often on VPN or corporate TLS inspection), the notebook sets `HF_HUB_DISABLE_SSL_VERIFY` (see `tourism_project/hf_http_config.py`). Prefer fixing trust with `SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE` when possible; GitHub Actions runners do not need this flag.

On **macOS**, local training needs OpenMP for XGBoost: `brew install libomp`. The notebook training cell sets `DYLD_LIBRARY_PATH` for the subprocess when Homebrew’s `opt/libomp/lib` exists; if `import xgboost` still fails in a plain terminal, run `brew link libomp --force` or export `DYLD_LIBRARY_PATH="$(brew --prefix)/opt/libomp/lib"`.
