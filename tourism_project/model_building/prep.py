"""
Load ``tourism.csv`` from the Hub, clean features, split train/test, save CSVs locally,
and upload the split files back to the same Dataset repository.

Parameters
----------
HF_TOKEN : str
    Required for uploading split files to the Hub.
HF_USER : str, optional
    Hub username for default dataset id (see ``data_register``).
HF_DATASET_REPO : str, optional
    Full dataset id; overrides default built from ``HF_USER``.
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

import pandas as pd
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

TARGET_COL = "ProdTaken"
DROP_COLS = {"CustomerID"}

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


def _hf_csv_uri(filename: str) -> str:
    """
    Build an ``hf://datasets/...`` URI for a file in the registered dataset.

    Parameters
    ----------
    filename : str
        File name inside the dataset repo (e.g. ``tourism.csv``).
    """
    return f"hf://datasets/{_dataset_repo_id()}/{filename}"


def load_raw_from_hub() -> pd.DataFrame:
    """Read ``tourism.csv`` from the Hub into a DataFrame."""
    path = _hf_csv_uri("tourism.csv")
    df = pd.read_csv(path)
    print(f"Loaded dataset from {path} shape={df.shape}.")
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop ID columns, impute missing values, and keep modeling columns plus target.

    Parameters
    ----------
    df : pd.DataFrame
        Raw tourism table from ``tourism.csv``.
    """
    out = df.copy()
    unnamed = [c for c in out.columns if c.startswith("Unnamed")]
    if unnamed:
        out = out.drop(columns=unnamed)
    out = out.drop(columns=[c for c in DROP_COLS if c in out.columns], errors="ignore")

    feature_cols = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in out.columns]
    missing = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COL]) - set(
        out.columns
    )
    if missing:
        raise ValueError(f"Dataset missing expected columns: {sorted(missing)}")

    for col in NUMERIC_FEATURES:
        med = out[col].median()
        out[col] = out[col].fillna(med)
    for col in CATEGORICAL_FEATURES:
        out[col] = out[col].fillna("Unknown").astype(str)

    out["Gender"] = out["Gender"].replace({"Fe Male": "Female"})

    out = out[feature_cols + [TARGET_COL]].dropna(subset=[TARGET_COL])
    out[TARGET_COL] = out[TARGET_COL].astype(int)
    return out


def main() -> None:
    """Run load, clean, stratified split, local save, and Hub upload of split CSVs."""
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable is required for uploads.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw_from_hub()
    clean = clean_dataframe(raw)
    print(f"After cleaning: shape={clean.shape}")

    y = clean[TARGET_COL]
    X = clean.drop(columns=[TARGET_COL])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    paths = {
        "Xtrain.csv": X_train,
        "Xtest.csv": X_test,
        "ytrain.csv": y_train,
        "ytest.csv": y_test,
    }
    for name, frame in paths.items():
        fp = DATA_DIR / name
        frame.to_csv(fp, index=False)
        print(f"Wrote {fp}")

    repo_id = _dataset_repo_id()
    api = HfApi(token=token)
    for name in paths:
        api.upload_file(
            path_or_fileobj=str(DATA_DIR / name),
            path_in_repo=name,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )
        print(f"Uploaded {name} to {repo_id}.")


if __name__ == "__main__":
    main()
