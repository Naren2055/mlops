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
import shutil
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
HF_CFG_SRC = ROOT / "hf_http_config.py"
HF_CFG_STAGING = DEPLOY_DIR / "hf_http_config.py"


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
    Create the Space if missing and upload the deployment folder in one commit.

    Stages ``tourism_project/hf_http_config.py`` into ``deployment/`` before upload so
    the Space repo root always contains ``hf_http_config.py`` alongside ``Dockerfile``
    (avoids a separate upload race and matches ``COPY . .`` in the image).

    Parameters
    ----------
    None
        Uses ``HF_TOKEN``, ``HF_USER`` or ``HF_SPACE_REPO`` from the environment.
    """
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable is required.")

    if not DEPLOY_DIR.is_dir():
        raise FileNotFoundError(f"Deployment folder not found: {DEPLOY_DIR}")

    if not (DEPLOY_DIR / "src" / "streamlit_app.py").is_file():
        raise FileNotFoundError(
            f"Missing Streamlit app: {DEPLOY_DIR / 'src' / 'streamlit_app.py'}"
        )
    if not (DEPLOY_DIR / "Dockerfile").is_file():
        raise FileNotFoundError(f"Missing Dockerfile in {DEPLOY_DIR}")

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

    copied_cfg = False
    if HF_CFG_SRC.is_file():
        shutil.copy2(HF_CFG_SRC, HF_CFG_STAGING)
        copied_cfg = True
    try:
        api.upload_folder(
            folder_path=str(DEPLOY_DIR),
            repo_id=repo_id,
            repo_type="space",
            path_in_repo="",
        )
    finally:
        if copied_cfg and HF_CFG_STAGING.is_file():
            HF_CFG_STAGING.unlink(missing_ok=True)

    print(f"Uploaded {DEPLOY_DIR} to Space {repo_id}.")


if __name__ == "__main__":
    main()
