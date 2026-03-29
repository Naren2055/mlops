"""
Wellness Tourism purchase prediction UI: collect features, load the trained pipeline
from the Hugging Face Model Hub, and display ``predict_proba`` for class 1.

The model is loaded lazily on **Predict** (or via **Load model only** in the sidebar).
Use **Clear model cache** after changing Space secrets or uploading a new joblib.

Parameters
----------
HF_MODEL_REPO : str, optional
    Model Hub id for ``hf_hub_download``; default ``snarendrababu41/wellness-tourism-xgboost-model``.
HF_MODEL_FILENAME : str, optional
    Artifact filename in that repo; default ``best_wellness_tourism_model.joblib``.
HF_TOKEN : str, optional
    Required when the model repo is private or gated (set as a Space repository secret).
HF_HUB_DISABLE_SSL_VERIFY : str, optional
    If ``hf_http_config.py`` is present at the app root, optional TLS override for Hub calls.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
for _root in (_script_dir.parent, _script_dir):
    _cfg = _root / "hf_http_config.py"
    if _cfg.is_file():
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        import hf_http_config

        hf_http_config.apply_hf_http_settings()
        break

import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

DEFAULT_MODEL_REPO = "snarendrababu41/wellness-tourism-xgboost-model"
DEFAULT_MODEL_FILE = "best_wellness_tourism_model.joblib"
CLASSIFICATION_THRESHOLD = 0.5


def _model_repo() -> str:
    """Return the effective model Hub id from ``HF_MODEL_REPO`` or the built-in default."""
    return os.environ.get("HF_MODEL_REPO", DEFAULT_MODEL_REPO).strip()


def _model_filename() -> str:
    """Return the joblib filename from ``HF_MODEL_FILENAME`` or the default."""
    return os.environ.get("HF_MODEL_FILENAME", DEFAULT_MODEL_FILE).strip()


@st.cache_resource(show_spinner="Downloading model from the Hub…")
def load_model(repo_id: str, filename: str):
    """
    Download and deserialize the joblib pipeline for the given Hub repo and file.

    Parameters
    ----------
    repo_id : str
        Hugging Face model repository id (``user/repo``).
    filename : str
        Path of the artifact inside that repository.

    Returns
    -------
    object
        Deserialized sklearn pipeline (supports ``predict_proba``).
    """
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    return joblib.load(path)


def _build_input_row(
    age: float,
    city_tier: int,
    duration_pitch: float,
    n_visiting: int,
    n_followups: float,
    product_pitched: str,
    pref_star: float,
    marital: str,
    n_trips: float,
    passport: int,
    pitch_score: int,
    own_car: int,
    n_children: float,
    designation: str,
    monthly_income: float,
    type_contact: str,
    occupation: str,
    gender: str,
) -> pd.DataFrame:
    """
    Assemble a single-row feature ``DataFrame`` matching training column names.

    Parameters
    ----------
    age : float
        Customer age in years.
    city_tier : int
        City tier (1–3).
    duration_pitch : float
        Pitch duration in minutes.
    n_visiting : int
        Number of persons visiting.
    n_followups : float
        Number of follow-ups.
    product_pitched : str
        Product category pitched.
    pref_star : float
        Preferred property star rating.
    marital : str
        Marital status label.
    n_trips : float
        Number of trips per year.
    passport : int
        Binary passport indicator (0/1).
    pitch_score : int
        Pitch satisfaction score (1–5).
    own_car : int
        Binary car ownership (0/1).
    n_children : float
        Children under 5 visiting.
    designation : str
        Job designation label.
    monthly_income : float
        Monthly income.
    type_contact : str
        Type of contact (e.g. Self Enquiry).
    occupation : str
        Occupation category.
    gender : str
        Gender label.

    Returns
    -------
    pandas.DataFrame
        One row with columns aligned to the training schema.
    """
    return pd.DataFrame(
        [
            {
                "Age": age,
                "CityTier": city_tier,
                "DurationOfPitch": duration_pitch,
                "NumberOfPersonVisiting": n_visiting,
                "NumberOfFollowups": n_followups,
                "PreferredPropertyStar": pref_star,
                "NumberOfTrips": n_trips,
                "Passport": passport,
                "PitchSatisfactionScore": pitch_score,
                "OwnCar": own_car,
                "NumberOfChildrenVisiting": n_children,
                "MonthlyIncome": monthly_income,
                "TypeofContact": type_contact,
                "Occupation": occupation,
                "Gender": gender,
                "ProductPitched": product_pitched,
                "MaritalStatus": marital,
                "Designation": designation,
            }
        ]
    )


def main() -> None:
    """
    Render the app: sidebar for testing helpers, form inputs, and prediction output.

    Parameters
    ----------
    None
        Reads environment variables and Streamlit widget state.
    """
    st.set_page_config(
        page_title="Wellness Tourism — Purchase prediction",
        page_icon="🧳",
        layout="wide",
    )

    repo = _model_repo()
    artifact = _model_filename()
    token_set = bool(os.environ.get("HF_TOKEN", "").strip())

    with st.sidebar:
        st.header("Testing")
        st.markdown(
            "Use this panel to confirm Hub settings and refresh the model after "
            "you change Space secrets or upload a new artifact."
        )
        st.text_input("Resolved `HF_MODEL_REPO`", value=repo, disabled=True)
        st.text_input("Resolved `HF_MODEL_FILENAME`", value=artifact, disabled=True)
        st.caption(f"`HF_TOKEN` secret: **{'set' if token_set else 'not set'}**")
        if st.button("Clear model cache", type="secondary"):
            load_model.clear()
            st.success("Cache cleared. Run **Predict** or **Load model only** again.")
        if st.button("Load model only", type="primary"):
            try:
                _ = load_model(repo, artifact)
                st.success("Model downloaded and loaded OK.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Load failed: {exc}")
        with st.expander("Checklist"):
            st.markdown(
                """
                - [ ] Space secret **`HF_MODEL_REPO`** = your `user/wellness-tourism-xgboost-model`
                - [ ] If repo is **private**: **`HF_TOKEN`** (read) is set on the Space
                - [ ] Artifact **`best_wellness_tourism_model.joblib`** exists on that repo
                - [ ] After changing secrets, click **Clear model cache**
                """
            )

    st.title("Wellness Tourism Package — Purchase prediction")
    st.write(
        "Estimate whether a customer is likely to purchase the **Wellness Tourism "
        "Package** (Visit with Us) from profile and pitch attributes."
    )

    left, right = st.columns((2, 1))
    with left:
        st.subheader("Inputs")
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", min_value=18.0, max_value=100.0, value=35.0)
            city_tier = st.selectbox("City tier", [1, 2, 3], index=0)
            duration_pitch = st.number_input(
                "Duration of pitch (minutes)", min_value=0.0, value=10.0
            )
            n_visiting = st.number_input(
                "Number of persons visiting", min_value=1, value=2
            )
            n_followups = st.number_input(
                "Number of follow-ups", min_value=0.0, value=3.0
            )
            product_pitched = st.selectbox(
                "Product pitched",
                ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"],
            )
            pref_star = st.number_input(
                "Preferred property star", min_value=1.0, max_value=5.0, value=3.0
            )
            marital = st.selectbox(
                "Marital status",
                ["Single", "Married", "Divorced", "Unmarried"],
            )
            n_trips = st.number_input(
                "Number of trips per year", min_value=0.0, value=2.0
            )
        with c2:
            passport = st.selectbox(
                "Passport", [0, 1], format_func=lambda x: "Yes" if x else "No"
            )
            pitch_score = st.slider("Pitch satisfaction score", 1, 5, 3)
            own_car = st.selectbox(
                "Owns car", [0, 1], format_func=lambda x: "Yes" if x else "No"
            )
            n_children = st.number_input(
                "Children under 5 visiting", min_value=0.0, value=0.0
            )
            designation = st.selectbox(
                "Designation",
                ["Executive", "Manager", "Senior Manager", "AVP", "VP"],
            )
            monthly_income = st.number_input(
                "Monthly income", min_value=0.0, value=20000.0
            )
            type_contact = st.selectbox(
                "Type of contact",
                ["Self Enquiry", "Company Invited"],
            )
            occupation = st.selectbox(
                "Occupation",
                ["Salaried", "Free Lancer", "Small Business", "Large Business"],
            )
            gender = st.selectbox("Gender", ["Male", "Female"])

        input_row = _build_input_row(
            age,
            city_tier,
            duration_pitch,
            n_visiting,
            n_followups,
            product_pitched,
            pref_star,
            marital,
            n_trips,
            passport,
            pitch_score,
            own_car,
            n_children,
            designation,
            monthly_income,
            type_contact,
            occupation,
            gender,
        )

        predict = st.button("Predict", type="primary")

    with right:
        st.subheader("Result")
        if predict:
            try:
                with st.spinner("Loading model and scoring…"):
                    model = load_model(repo, artifact)
                    proba = float(model.predict_proba(input_row)[0, 1])
            except Exception as exc:  # noqa: BLE001
                st.error("Prediction failed.")
                st.code(str(exc), language="text")
                with st.expander("Full traceback (for debugging)"):
                    st.exception(exc)
                st.info(
                    "Verify Space secrets **`HF_MODEL_REPO`** / **`HF_TOKEN`**, "
                    "then **Clear model cache** and retry."
                )
            else:
                pred = int(proba >= CLASSIFICATION_THRESHOLD)
                label = "Likely buyer (1)" if pred else "Unlikely buyer (0)"
                st.metric("P(purchase)", f"{proba:.2%}")
                st.success(
                    f"Predicted class: **{label}** (threshold {CLASSIFICATION_THRESHOLD})."
                )
        else:
            st.caption("Adjust inputs and click **Predict**.")

        with st.expander("Feature row (debug)"):
            st.dataframe(input_row, use_container_width=True)


if __name__ == "__main__":
    main()
