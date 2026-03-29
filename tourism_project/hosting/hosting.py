"""
Upload ``tourism_project/deployment`` to a Hugging Face Space (Docker SDK) so the
Streamlit image can build from the Dockerfile.

Parameters
----------
HF_TOKEN : str
    Token with write access to Spaces (required).
HF_USER : str, optional
    Username; default Space id ``{HF_USER}/wellness-tourism-streamlit``.
HF_SPACE_REPO : str, optional
    Full Space id ``user/space-name``; overrides default built from ``HF_USER``.
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

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

ROOT = Path(__file__).resolve().parents[1]
DEPLOY_DIR = ROOT / "deployment"


def _space_repo_id() -> str:
    """Resolve the Space repository id from ``HF_SPACE_REPO`` or ``HF_USER``."""
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
    Create the Space if missing, upload the deployment folder, and upload ``hf_http_config.py``
    beside ``app.py`` when present so Streamlit can apply optional TLS settings.
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
    hf_cfg = ROOT / "hf_http_config.py"
    if hf_cfg.is_file():
        api.upload_file(
            path_or_fileobj=str(hf_cfg),
            path_in_repo="hf_http_config.py",
            repo_id=repo_id,
            repo_type="space",
            token=token,
        )
        print(f"Uploaded {hf_cfg.name} to Space root.")
    print(f"Uploaded {DEPLOY_DIR} to Space {repo_id}.")


if __name__ == "__main__":
    main()
