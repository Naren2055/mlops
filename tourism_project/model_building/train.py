"""
Tune an XGBoost pipeline on Hub-hosted train/test CSVs, log runs to MLflow, and
upload the best ``joblib`` model to the Hugging Face Model Hub.

Parameters
----------
HF_TOKEN : str
    Required for creating the model repo and uploading the artifact.
HF_USER : str, optional
    Username for default dataset and model repo ids when overrides are not set.
HF_DATASET_REPO : str, optional
    Dataset id containing ``Xtrain.csv``, ``Xtest.csv``, ``ytrain.csv``, ``ytest.csv``.
HF_MODEL_REPO : str, optional
    Model id for the uploaded ``joblib`` file; default ``{HF_USER}/wellness-tourism-xgboost-model``.

Notes
-----
On **macOS**, the pip XGBoost wheel needs OpenMP (``libomp.dylib``). Run ``brew install libomp``.
The formula is *keg-only*, so XGBoost often cannot find the library unless you also run
``brew link libomp --force`` (or start Python with ``DYLD_LIBRARY_PATH`` set to
``$(brew --prefix)/opt/libomp/lib``). Linux CI images typically already ship compatible OpenMP.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_tp = Path(__file__).resolve().parents[1]
if str(_tp) not in sys.path:
    sys.path.insert(0, str(_tp))
import hf_http_config

hf_http_config.apply_hf_http_settings()

import joblib
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
from sklearn.compose import make_column_transformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "Age",
    "CityTier",
    "DurationOfPitch",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "MonthlyIncome",
]
CATEGORICAL_FEATURES = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
    "Designation",
]

ROOT = Path(__file__).resolve().parents[1]
MLRUNS_DIR = ROOT / "mlruns"
MODEL_FILENAME = "best_wellness_tourism_model.joblib"


def _dataset_repo_id() -> str:
    """Resolve the dataset repository id from ``HF_DATASET_REPO`` or ``HF_USER``."""
    explicit = os.getenv("HF_DATASET_REPO")
    if explicit:
        return explicit.strip()
    user = os.getenv("HF_USER", "").strip()
    if not user:
        raise ValueError(
            "Set HF_USER (username) or HF_DATASET_REPO (full dataset repo id)."
        )
    return f"{user}/wellness-tourism-purchase"


def _model_repo_id() -> str:
    """Resolve the model repository id from ``HF_MODEL_REPO`` or ``HF_USER``."""
    explicit = os.getenv("HF_MODEL_REPO")
    if explicit:
        return explicit.strip()
    user = os.getenv("HF_USER", "").strip()
    if not user:
        raise ValueError(
            "Set HF_USER (username) or HF_MODEL_REPO (full model repo id)."
        )
    return f"{user}/wellness-tourism-xgboost-model"


def _load_xy() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Load train/test feature matrices and labels from the Hub via ``hf://`` URIs."""
    base = _dataset_repo_id()
    X_train = pd.read_csv(f"hf://datasets/{base}/Xtrain.csv")
    X_test = pd.read_csv(f"hf://datasets/{base}/Xtest.csv")
    y_train = pd.read_csv(f"hf://datasets/{base}/ytrain.csv").squeeze("columns")
    y_test = pd.read_csv(f"hf://datasets/{base}/ytest.csv").squeeze("columns")
    y_train = np.asarray(y_train).ravel()
    y_test = np.asarray(y_test).ravel()
    return X_train, X_test, y_train, y_test


def main() -> None:
    """Run grid search, MLflow logging, evaluation, ``joblib`` export, and Hub upload."""
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable is required.")

    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file:{MLRUNS_DIR}")
    mlflow.set_experiment("wellness-tourism-xgboost")

    X_train, X_test, y_train, y_test = _load_xy()
    print(f"Train shape={X_train.shape}, Test shape={X_test.shape}")

    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale_pos_weight = float(neg) / float(pos) if pos > 0 else 1.0

    preprocessor = make_column_transformer(
        (StandardScaler(), NUMERIC_FEATURES),
        (OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    )
    xgb_model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
    )
    model_pipeline = make_pipeline(preprocessor, xgb_model)

    param_grid = {
        "xgbclassifier__n_estimators": [100],
        "xgbclassifier__max_depth": [3, 5],
        "xgbclassifier__learning_rate": [0.05, 0.1],
        "xgbclassifier__colsample_bytree": [0.8],
        "xgbclassifier__reg_lambda": [0.5, 1.0],
    }

    api = HfApi(token=token)
    model_repo = _model_repo_id()

    with mlflow.start_run(run_name="wellness_tourism_grid_search"):
        grid_search = GridSearchCV(
            model_pipeline,
            param_grid,
            cv=5,
            n_jobs=-1,
            scoring="f1",
        )
        grid_search.fit(X_train, y_train)

        results = grid_search.cv_results_
        for i in range(len(results["params"])):
            with mlflow.start_run(nested=True):
                mlflow.log_params(results["params"][i])
                mlflow.log_metric("mean_test_score", float(results["mean_test_score"][i]))
                mlflow.log_metric("std_test_score", float(results["std_test_score"][i]))

        mlflow.log_params(grid_search.best_params_)

        best_model = grid_search.best_estimator_
        threshold = 0.5

        y_pred_train = (best_model.predict_proba(X_train)[:, 1] >= threshold).astype(int)
        y_pred_test = (best_model.predict_proba(X_test)[:, 1] >= threshold).astype(int)

        train_report = classification_report(
            y_train, y_pred_train, output_dict=True, zero_division=0
        )
        test_report = classification_report(
            y_test, y_pred_test, output_dict=True, zero_division=0
        )

        mlflow.log_metrics(
            {
                "train_accuracy": train_report["accuracy"],
                "train_precision_class1": train_report["1"]["precision"],
                "train_recall_class1": train_report["1"]["recall"],
                "train_f1_class1": train_report["1"]["f1-score"],
                "test_accuracy": test_report["accuracy"],
                "test_precision_class1": test_report["1"]["precision"],
                "test_recall_class1": test_report["1"]["recall"],
                "test_f1_class1": test_report["1"]["f1-score"],
            }
        )

        print("Best params:", grid_search.best_params_)
        print("Train report:\n", classification_report(y_train, y_pred_train))
        print("Test report:\n", classification_report(y_test, y_pred_test))

        out_path = Path(MODEL_FILENAME)
        joblib.dump(best_model, out_path)
        mlflow.log_artifact(str(out_path), artifact_path="model")

        try:
            api.repo_info(repo_id=model_repo, repo_type="model")
            print(f"Model repo '{model_repo}' exists.")
        except RepositoryNotFoundError:
            print(f"Creating public model repo '{model_repo}'...")
            create_repo(repo_id=model_repo, repo_type="model", private=False, token=token)

        api.upload_file(
            path_or_fileobj=str(out_path),
            path_in_repo=MODEL_FILENAME,
            repo_id=model_repo,
            repo_type="model",
            token=token,
        )
        print(f"Uploaded {MODEL_FILENAME} to {model_repo}.")


if __name__ == "__main__":
    main()
