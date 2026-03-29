"""
Streamlit UI for Wellness Tourism package purchase prediction: collect inputs,
build one feature row, load the trained pipeline from the Hub, and show ``predict_proba``.

Runs on port **8501** in the container, same as the case study (see ``README.md``
``app_port``). The model loads on the first **Predict** click.

Parameters
----------
HF_MODEL_REPO : str, optional
    Model Hub id for ``hf_hub_download``; default ``snarendrababu42/wellness-tourism-xgboost-model``.
HF_MODEL_FILENAME : str, optional
    Artifact name inside that repo; default ``best_wellness_tourism_model.joblib``.
HF_TOKEN : str, optional
    Set on the Space when the model repo is private or gated.
HF_HUB_DISABLE_SSL_VERIFY : str, optional
    Same as ``tourism_project/hf_http_config``; rarely needed on the Space.
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

MODEL_REPO = os.environ.get(
    "HF_MODEL_REPO",
    "snarendrababu42/wellness-tourism-xgboost-model",
)
MODEL_FILE = os.environ.get("HF_MODEL_FILENAME", "best_wellness_tourism_model.joblib")
CLASSIFICATION_THRESHOLD = 0.5


@st.cache_resource
def load_model():
    """
    Download and deserialize the ``joblib`` pipeline using ``MODEL_REPO`` and ``MODEL_FILE``.
    """
    path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    return joblib.load(path)


def main() -> None:
    """Render widgets, build ``input_row``, and on *Predict* run ``load_model`` and inference."""
    st.title("Wellness Tourism Package — Purchase Prediction")
    st.write(
        "Predict whether a customer is likely to purchase the **Wellness Tourism "
        "Package** (Visit with Us) from profile and sales-interaction attributes."
    )
    st.caption(
        f"Model repo: `{MODEL_REPO}` · file: `{MODEL_FILE}` · "
        "The first *Predict* may take longer while the artifact downloads."
    )

    st.subheader("Customer and interaction inputs")

    age = st.number_input("Age", min_value=18.0, max_value=100.0, value=35.0)
    city_tier = st.selectbox("City tier", [1, 2, 3], index=0)
    duration_pitch = st.number_input(
        "Duration of pitch (minutes)", min_value=0.0, value=10.0
    )
    n_visiting = st.number_input("Number of persons visiting", min_value=1, value=2)
    n_followups = st.number_input("Number of follow-ups", min_value=0.0, value=3.0)
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
    n_trips = st.number_input("Number of trips per year", min_value=0.0, value=2.0)
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
    monthly_income = st.number_input("Monthly income", min_value=0.0, value=20000.0)
    type_contact = st.selectbox(
        "Type of contact",
        ["Self Enquiry", "Company Invited"],
    )
    occupation = st.selectbox(
        "Occupation",
        ["Salaried", "Free Lancer", "Small Business", "Large Business"],
    )
    gender = st.selectbox("Gender", ["Male", "Female"])

    input_row = pd.DataFrame(
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

    if st.button("Predict"):
        try:
            with st.spinner(
                "Loading model and predicting (first run downloads from the Hub)..."
            ):
                model = load_model()
                proba = float(model.predict_proba(input_row)[0, 1])
        except Exception as exc:  # noqa: BLE001
            st.error(
                f"Could not load or run the model from `{MODEL_REPO}` / `{MODEL_FILE}`. "
                "Set Space secret `HF_MODEL_REPO` (and `HF_TOKEN` if the repo is private). "
                f"Details: {exc}"
            )
        else:
            pred = int(proba >= CLASSIFICATION_THRESHOLD)
            label = "Likely buyer (1)" if pred else "Unlikely buyer (0)"
            st.success(
                f"Estimated probability of purchase: **{proba:.2%}** — "
                f"predicted class: **{label}**."
            )


main()
