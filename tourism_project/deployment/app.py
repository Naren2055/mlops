"""
Streamlit frontend for the Wellness Tourism package purchase predictor.

Loads the trained sklearn pipeline (preprocessor + XGBoost) from the Hugging Face
Model Hub, collects user inputs aligned with training features, and displays
predicted probability of purchase.
"""

from __future__ import annotations

import os

import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

# Default model artifact; override with env for different Hub users or filenames.
MODEL_REPO = os.environ.get(
    "HF_MODEL_REPO",
    "snarendrababu42/wellness-tourism-xgboost-model",
)
MODEL_FILE = os.environ.get("HF_MODEL_FILENAME", "best_wellness_tourism_model.joblib")


@st.cache_resource
def load_model():
    """
    Download and deserialize the joblib pipeline from the Model Hub.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Fitted preprocessor + classifier pipeline.

    Notes
    -----
    Uses ``hf_hub_download`` so the Space does not need the model committed in git.
    """
    path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    return joblib.load(path)


def main() -> None:
    """Render the Streamlit UI and run inference on button click."""
    st.title("Wellness Tourism Package — Purchase Prediction")
    st.write(
        "This app predicts whether a customer is likely to purchase the "
        "**Wellness Tourism Package** (Visit with Us), based on profile and "
        "sales-interaction attributes."
    )

    try:
        model = load_model()
    except Exception as exc:  # noqa: BLE001 — show friendly message in UI
        st.error(
            f"Could not load model from `{MODEL_REPO}`. "
            f"Set Space secrets / variables `HF_MODEL_REPO` and ensure the file "
            f"`{MODEL_FILE}` exists. Details: {exc}"
        )
        st.stop()

    st.subheader("Customer & interaction inputs")

    age = st.number_input("Age", min_value=18.0, max_value=100.0, value=35.0)
    city_tier = st.selectbox("City tier", [1, 2, 3], index=0)
    duration_pitch = st.number_input("Duration of pitch (minutes)", min_value=0.0, value=10.0)
    n_visiting = st.number_input("Number of persons visiting", min_value=1, value=2)
    n_followups = st.number_input("Number of follow-ups", min_value=0.0, value=3.0)
    product_pitched = st.selectbox(
        "Product pitched",
        ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"],
    )
    pref_star = st.number_input("Preferred property star", min_value=1.0, max_value=5.0, value=3.0)
    marital = st.selectbox(
        "Marital status",
        ["Single", "Married", "Divorced", "Unmarried"],
    )
    n_trips = st.number_input("Number of trips per year", min_value=0.0, value=2.0)
    passport = st.selectbox("Passport", [0, 1], format_func=lambda x: "Yes" if x else "No")
    pitch_score = st.slider("Pitch satisfaction score", 1, 5, 3)
    own_car = st.selectbox("Owns car", [0, 1], format_func=lambda x: "Yes" if x else "No")
    n_children = st.number_input("Children under 5 visiting", min_value=0.0, value=0.0)
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

    row = pd.DataFrame(
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
        proba = float(model.predict_proba(row)[0, 1])
        pred = int(proba >= 0.5)
        st.success(
            f"Estimated probability of purchase: **{proba:.2%}** — "
            f"predicted class: **{'Likely buyer (1)' if pred else 'Unlikely buyer (0)'}**."
        )


if __name__ == "__main__":
    main()
