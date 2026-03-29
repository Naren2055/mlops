"""
Optional Hugging Face Hub HTTP/TLS settings for environments where default verification fails.

Typical cause: corporate VPN or TLS inspection presents a self-signed intermediate; Python
then raises ``SSLCertVerificationError``. Prefer fixing trust (``SSL_CERT_FILE`` or
``REQUESTS_CA_BUNDLE`` pointing at your organization's CA bundle). This module supports an
explicit opt-in to skip verification for local development only.

Parameters
----------
HF_HUB_DISABLE_SSL_VERIFY : str, optional
    When ``1``, ``true``, ``yes``, or ``on`` (case-insensitive), Hub HTTP calls in this
    process use a ``requests.Session`` with ``verify=False``. **Insecure** (traffic can be
    intercepted); do not use in production or on shared machines unless policy allows.
"""

from __future__ import annotations

import os


def _ssl_verify_disabled() -> bool:
    """
    Return whether ``HF_HUB_DISABLE_SSL_VERIFY`` requests disabling TLS verification.

    Parameters
    ----------
    None
        Reads only the process environment.
    """
    flag = os.environ.get("HF_HUB_DISABLE_SSL_VERIFY", "").strip().lower()
    return flag in ("1", "true", "yes", "on")


def apply_hf_http_settings() -> None:
    """
    Register a custom Hugging Face Hub HTTP backend when SSL verify should be skipped.

    No-op unless ``HF_HUB_DISABLE_SSL_VERIFY`` is truthy. Must run before any Hub API
    calls in the process (call at startup of CLI scripts and apps).

    Parameters
    ----------
    None
        Controlled by ``HF_HUB_DISABLE_SSL_VERIFY`` in the environment.
    """
    if not _ssl_verify_disabled():
        return

    import urllib3
    import requests
    from huggingface_hub import configure_http_backend

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def backend_factory() -> requests.Session:
        """Build a session with TLS verification disabled for Hub requests."""
        session = requests.Session()
        session.verify = False
        return session

    configure_http_backend(backend_factory=backend_factory)
