"""
Push deployment assets to a Hugging Face Space (Docker / Streamlit frontend).

Uploads every file under ``tourism_project/deployment`` to the configured Space
repository so the public app can rebuild from the Dockerfile.

Environment variables
---------------------
HF_TOKEN : str
    Required. Token with write access to Spaces.
HF_SPACE_REPO : str, optional
    Full Space repo id, default ``{HF_USER}/wellness-tourism-streamlit``.
HF_USER : str
    Required if ``HF_SPACE_REPO`` is not set.
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

ROOT = Path(__file__).resolve().parents[1]
DEPLOY_DIR = ROOT / "deployment"


def _space_repo_id() -> str:
    """
    Resolve the Hugging Face Space repository id.

    Returns
    -------
    str
        Space repo id ``username/space-name``.
    """
    explicit = os.getenv("HF_SPACE_REPO")
    if explicit:
        return explicit.strip()
    user = os.getenv("HF_USER", "").strip()
    if not user:
        raise ValueError(
            "Set HF_USER (username) or HF_SPACE_REPO (full Space repo id)."
        )
    return f"{user}/wellness-tourism-streamlit"


def main() -> None:
    """
    Upload the deployment folder to the Hugging Face Space.
    """
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable is required.")

    if not DEPLOY_DIR.is_dir():
        raise FileNotFoundError(f"Deployment folder not found: {DEPLOY_DIR}")

    repo_id = _space_repo_id()
    api = HfApi(token=token)
    try:
        api.repo_info(repo_id=repo_id, repo_type="space")
    except RepositoryNotFoundError:
        create_repo(
            repo_id=repo_id,
            repo_type="space",
            private=False,
            token=token,
            space_sdk="docker",
        )
        print(f"Created Space {repo_id} (docker). Configure SDK in Hub UI if needed.")

    api.upload_folder(
        folder_path=str(DEPLOY_DIR),
        repo_id=repo_id,
        repo_type="space",
        path_in_repo="",
    )
    print(f"Uploaded {DEPLOY_DIR} to Space {repo_id}.")


if __name__ == "__main__":
    main()
