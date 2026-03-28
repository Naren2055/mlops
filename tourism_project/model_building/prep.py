"""
Data preparation for the Wellness Tourism purchase prediction project.

Loads the raw CSV from the Hugging Face dataset hub, cleans rows and columns,
splits into train/test sets, saves CSV files locally under ``tourism_project/data``,
and uploads those splits back to the same dataset repository.

Environment variables
---------------------
HF_TOKEN : str
    Required for uploads.
HF_USER or HF_DATASET_REPO : str
    Same convention as ``data_register.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

TARGET_COL = "ProdTaken"
DROP_COLS = {"CustomerID"}

# Features used for modeling (all meaningful columns except target and IDs).
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
    """Return Hugging Face dataset repo id (see ``data_register``)."""
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
    """Build ``hf://datasets/...`` URI for a file in the registered dataset."""
    return f"hf://datasets/{_dataset_repo_id()}/{filename}"


def load_raw_from_hub() -> pd.DataFrame:
    """
    Load ``tourism.csv`` from the Hugging Face dataset repository.

    Returns
    -------
    pd.DataFrame
        Raw table as stored on the Hub.
    """
    path = _hf_csv_uri("tourism.csv")
    df = pd.read_csv(path)
    print(f"Loaded dataset from {path} shape={df.shape}.")
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop identifiers, fix obvious data issues, and impute missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Raw tourism table.

    Returns
    -------
    pd.DataFrame
        Cleaned table with only modeling columns (features + target).
    """
    out = df.copy()
    # Strip index column if present from CSV export
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

    # Fix common data-entry typo so encodings align with deployment UI options.
    out["Gender"] = out["Gender"].replace({"Fe Male": "Female"})

    out = out[feature_cols + [TARGET_COL]].dropna(subset=[TARGET_COL])
    out[TARGET_COL] = out[TARGET_COL].astype(int)
    return out


def main() -> None:
    """
    Run full prep: load from Hub, clean, split, save locally, upload splits.
    """
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
