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
    app.py                       # Streamlit; loads model from HF Model Hub
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

## Space configuration

In the Space **Settings → Repository secrets**, set `HF_MODEL_REPO` to your model repo (e.g. `username/wellness-tourism-xgboost-model`) so `app.py` can download `best_wellness_tourism_model.joblib`.

## Local / notebook

Set `HF_TOKEN` and `HF_USER`, then run the notebook **sequentially** (or run the Python scripts under `tourism_project/model_building/` and `tourism_project/hosting/`).
