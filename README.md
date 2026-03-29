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
    README.md                    # YAML: sdk docker, app_port 8501 (case study port)
    app.py                       # Streamlit; Hub download + DataFrame + predict_proba
    requirements.txt
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
- **Port:** `deployment/Dockerfile` and `deployment/README.md` both use **`8501`**, matching the **Case_Study_MLOps** notebooks. `app_port` in `README.md` must match `--server.port` in the `Dockerfile` or the Space can stay stuck on **Starting**.
- **Layout:** After `hosting.py`, the Space repo root must contain **`README.md`**, **`Dockerfile`**, **`app.py`**, **`requirements.txt`**.
- **Secrets:** Space **Repository secrets** — `HF_MODEL_REPO` (and `HF_TOKEN` if the model repo is private).

## Local / notebook

Set `HF_TOKEN` and `HF_USER`, then run the notebook **sequentially** (or run the Python scripts under `tourism_project/model_building/` and `tourism_project/hosting/`).
