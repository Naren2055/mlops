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

### Docker Space (default in this repo)

- **Port:** Hugging Face’s Docker default is **`app_port: 7860`**. The Dockerfile runs Streamlit on **7860** and `deployment/README.md` sets the same `app_port`. If the app listens on 8501 while the platform expects 7860, the Space often stays **“Starting”** forever after a successful build.
- **`README.md` must be at the root of the Space repository** (same level as `Dockerfile` and `app.py`). `hosting.py` uploads the whole `deployment/` folder, which is correct.
- **Secrets:** set **`HF_MODEL_REPO`** (and **`HF_TOKEN`** if the model is private) under Space **Settings → Repository secrets**.

### Alternative: native Streamlit Space (no Docker)

If Docker keeps failing provisioning, use Hugging Face’s **Streamlit SDK** (HF runs Streamlit for you on port **8501**; no `Dockerfile`).

1. On the Space: **Settings → Change Space SDK** → choose **Streamlit** (or create a new Space with SDK **Streamlit**).
2. In the Space repo, keep only **`app.py`**, **`requirements.txt`**, and **`README.md`** with a YAML block like:

   ```yaml
   ---
   title: Wellness Tourism Purchase Predictor
   sdk: streamlit
   sdk_version: 1.43.2
   ---
   ```

3. **Remove** `Dockerfile` from that Space repo (Streamlit Spaces should not use a custom Docker image unless you know you need it).
4. Rebuild / open the App tab.

You still need the same **Repository secrets** (`HF_MODEL_REPO`, etc.). The **course rubric** asks for a Dockerfile in the project: keep `tourism_project/deployment/Dockerfile` in **GitHub** even if you deploy the Space using the Streamlit SDK for reliability.

## Local / notebook

Set `HF_TOKEN` and `HF_USER`, then run the notebook **sequentially** (or run the Python scripts under `tourism_project/model_building/` and `tourism_project/hosting/`).
