"""
Register raw tourism data on the Hugging Face Hub (dataset repository).

This script creates the dataset repo if missing and uploads every file under
``tourism_project/data`` (for example ``tourism.csv``). It expects:

- ``HF_TOKEN``: Hugging Face API token with write access.
- ``HF_USER``: Your Hugging Face username (or set ``HF_DATASET_REPO`` to
  ``username/repo-name`` explicitly).

Environment variables
---------------------
HF_TOKEN : str
    Required. Token for ``huggingface_hub`` authentication.
HF_USER : str, optional
    Hub username; dataset id becomes ``{HF_USER}/wellness-tourism-purchase``.
HF_DATASET_REPO : str, optional
    Full dataset repo id; overrides the default built from ``HF_USER``.
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# Project root: tourism_project/
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def _dataset_repo_id() -> str:
    """
    Resolve the target Hugging Face dataset repository id.

    Returns
    -------
    str
        Dataset repo id in the form ``username/repo-name``.

    Raises
    ------
    ValueError
        If neither ``HF_DATASET_REPO`` nor ``HF_USER`` is set.
    """
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
    """
    Ensure the dataset repository exists and upload the local ``data`` folder.

    Raises
    ------
    FileNotFoundError
        If ``tourism_project/data`` does not exist or contains no files.
    """
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
