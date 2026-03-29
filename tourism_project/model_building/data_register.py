"""
Parameters
----------
HF_TOKEN : str
    Hugging Face API token with write access (required).
HF_USER : str, optional
    Hub username; target repo becomes ``{HF_USER}/wellness-tourism-purchase``.
HF_DATASET_REPO : str, optional
    Full dataset id ``user/repo``; overrides the default built from ``HF_USER``.
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


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


def main() -> None:
    """Ensure the Hub dataset exists and upload ``tourism_project/data`` to it."""
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable is required for uploads.")

    repo_id = _dataset_repo_id()
    repo_type = "dataset"
    api = HfApi(token=token)

    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Dataset repo '{repo_id}' already exists. Uploading files.")
    except RepositoryNotFoundError:
        print(f"Dataset repo '{repo_id}' not found. Creating public repo...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False, token=token)
        print(f"Dataset repo '{repo_id}' created.")

    api.upload_folder(
        folder_path=str(DATA_DIR),
        repo_id=repo_id,
        repo_type=repo_type,
        token=token,
    )
    print(f"Uploaded contents of {DATA_DIR} to {repo_id}.")


if __name__ == "__main__":
    main()
