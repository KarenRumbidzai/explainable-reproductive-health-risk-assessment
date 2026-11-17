"""
model_service.py

Everything related to the KRM model lives here, separate from Flask's
routing/HTTP concerns in app.py. This module:

  1. Loads the saved KRM pipeline artifacts once, at import time.
  2. Replicates the exact feature-engineering logic used when the model
     was trained (must stay in sync with the training notebook's
     encoding if the model is ever retrained).
  3. Runs a prediction + SHAP explanation for one respondent.
  4. Maps risk factors to personalized, non-medical suggestions.

Kept independent of Flask, sqlite, or Excel I/O so it can be unit tested
or reused (e.g. in a batch scoring script) without pulling in the web
framework.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

MODELS_DIR = Path(__file__).resolve().parent / "models"

CLASS_NAMES = ["Low", "Moderate", "High"]

RECOMMENDATION_LIBRARY: dict[str, str] = {
    "irregular_menstrual_cycle": "Consider discussing your cycle pattern with a healthcare provider; menstrual irregularity can be an early indicator worth medical follow-up.",
    "severe_menstrual_pain": "Persistent severe menstrual pain is worth raising with a healthcare provider, as it can be associated with several reproductive conditions.",
    "bmi_high": "Gradual, sustainable changes toward a healthier weight range (balanced diet, regular activity) may help reduce reproductive risk.",
    "bmi_low": "Being underweight can also affect reproductive health; a nutrition-focused check-in with a healthcare provider may help.",
    "pesticide_exposure": "Where possible, reduce direct contact with pesticides (protective equipment, handwashing after exposure).",
    "air_pollution_exposure": "Reducing prolonged exposure to smoke/polluted air where practical may help lower environmental risk.",
    "occupational_chemical_exposure": "Where possible, discuss protective measures for on-the-job chemical exposure with your employer or a health provider.",
    "smoking": "Reducing or stopping smoking is one of the most impactful changes available for reproductive and general health.",
    "physical_activity": "Increasing regular moderate physical activity is associated with better reproductive/metabolic health.",
    "sleep_hours": "Aiming for more consistent, adequate sleep may support hormonal and reproductive health.",
    "stress_level": "Stress-reduction practices (rest, social support, counselling if needed) may help, since stress has been linked to menstrual and reproductive irregularities.",
    "fruit_vegetable_frequency_1_5": "Increasing fruit and vegetable intake can improve overall dietary quality linked to reproductive health.",
    "processed_food_frequency_1_5": "Reducing frequency of processed food may improve overall dietary quality.",
}

DISCLAIMER = (
    "This tool produces an automated reproductive health RISK ESTIMATE for "
    "decision-support purposes only. It is NOT a medical diagnosis and does "
    "NOT replace professional medical advice. Please consult a qualified "
    "healthcare provider for any reproductive health concerns."
)


@dataclass(frozen=True)
class ModelBundle:
    """Everything loaded once at startup and reused across requests."""
    krm_model: Any
    xgb_model: Any
    scaler: Any
    feature_cols: list[str]
    ordinal_maps: dict[str, dict[str, int]]
    binary_maps: dict[str, dict[str, int]]
    provinces: list[str]
    numeric_cols_to_scale: list[str]
    explainer: Any = field(repr=False)

    @property
    def form_options(self) -> dict[str, list[str]]:
        """Dropdown choices for the prediction form, derived directly
        from the encoding maps used to train the model -- guarantees the
        form can never submit a category the model wasn't trained on."""
        return {
            "province": self.provinces,
            "residence": list(self.binary_maps["residence"].keys()),
            "physical_activity": list(self.ordinal_maps["physical_activity"].keys()),
            "stress_level": list(self.ordinal_maps["stress_level"].keys()),
            "pesticide_exposure": list(self.ordinal_maps["pesticide_exposure"].keys()),
            "air_pollution_exposure": list(self.ordinal_maps["air_pollution_exposure"].keys()),
            "alcohol_use": list(self.ordinal_maps["alcohol_use"].keys()),
            "occupational_chemical_exposure": list(self.binary_maps["occupational_chemical_exposure"].keys()),
            "smoking": list(self.binary_maps["smoking"].keys()),
            "irregular_menstrual_cycle": list(self.binary_maps["irregular_menstrual_cycle"].keys()),
            "severe_menstrual_pain": list(self.binary_maps["severe_menstrual_pain"].keys()),
            "family_reproductive_history": list(self.binary_maps["family_reproductive_history"].keys()),
            "previous_reproductive_condition": list(self.binary_maps["previous_reproductive_condition"].keys()),
        }


def load_model_bundle(models_dir: Path = MODELS_DIR) -> ModelBundle:
    """Load every saved artifact once. Raises a clear error early if
    anything is missing, rather than failing deep inside a request."""
    required = [
        "krm_model.joblib", "xgboost.joblib", "scaler.joblib",
        "feature_columns.joblib", "encoding_maps.json",
    ]
    missing = [f for f in required if not (models_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing model artifact(s) in {models_dir}: {missing}. "
            "Run the training notebook's 'Saving the Final Model' section first."
        )

    krm_model = joblib.load(models_dir / "krm_model.joblib")
    xgb_model = joblib.load(models_dir / "xgboost.joblib")
    scaler = joblib.load(models_dir / "scaler.joblib")
    feature_cols = joblib.load(models_dir / "feature_columns.joblib")

    with open(models_dir / "encoding_maps.json") as f:
        encoding_maps = json.load(f)

    # TreeExplainer only needs the tree model itself, so it's cheap to
    # rebuild fresh here rather than depend on a separately serialized
    # (and separately version-sensitive) explainer file.
    explainer = shap.TreeExplainer(xgb_model)

    return ModelBundle(
        krm_model=krm_model,
        xgb_model=xgb_model,
        scaler=scaler,
        feature_cols=feature_cols,
        ordinal_maps=encoding_maps["ordinal_maps"],
        binary_maps=encoding_maps["binary_maps"],
        provinces=encoding_maps["provinces"],
        numeric_cols_to_scale=encoding_maps["numeric_cols_to_scale"],
        explainer=explainer,
    )


def bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    return "Obese"


def engineer_single_respondent(raw: dict, bundle: ModelBundle) -> pd.DataFrame:
    """Mirror the exact feature engineering used at training time for one
    new respondent's raw questionnaire answers."""
    row: dict[str, Any] = {
        "age": raw["age"],
        "processed_food_frequency_1_5": raw["processed_food_frequency_1_5"],
        "fruit_vegetable_frequency_1_5": raw["fruit_vegetable_frequency_1_5"],
        "legume_frequency_1_5": raw["legume_frequency_1_5"],
        "sleep_hours": raw["sleep_hours"],
        "bmi": raw["bmi"],
    }
    for col, mapping in bundle.ordinal_maps.items():
        row[col] = mapping[raw[col]]
    for col, mapping in bundle.binary_maps.items():
        row[col] = mapping[raw[col]]

    cat = bmi_category(raw["bmi"])
    row["bmi_cat_Obese"] = int(cat == "Obese")
    row["bmi_cat_Overweight"] = int(cat == "Overweight")
    row["bmi_cat_Underweight"] = int(cat == "Underweight")

    for p in bundle.provinces:
        if p == "Bulawayo":
            continue  # reference category (drop_first=True at training time)
        row[f"province_{p}"] = int(raw["province"] == p)

    return pd.DataFrame([row]).reindex(columns=bundle.feature_cols, fill_value=0)


def generate_recommendations(raw: dict) -> list[str]:
    """Map elevated risk factors present in this respondent's answers to
    plain-language, non-medical suggestions. Capped at 4 so the result
    stays focused rather than overwhelming."""
    recs: list[str] = []
    if raw.get("irregular_menstrual_cycle") == "Yes":
        recs.append(RECOMMENDATION_LIBRARY["irregular_menstrual_cycle"])
    if raw.get("severe_menstrual_pain") == "Yes":
        recs.append(RECOMMENDATION_LIBRARY["severe_menstrual_pain"])
    if raw.get("bmi", 0) >= 30:
        recs.append(RECOMMENDATION_LIBRARY["bmi_high"])
    elif raw.get("bmi", 100) < 18.5:
        recs.append(RECOMMENDATION_LIBRARY["bmi_low"])
    if raw.get("pesticide_exposure") in ("Moderate", "High"):
        recs.append(RECOMMENDATION_LIBRARY["pesticide_exposure"])
    if raw.get("air_pollution_exposure") == "High":
        recs.append(RECOMMENDATION_LIBRARY["air_pollution_exposure"])
    if raw.get("occupational_chemical_exposure") == "Yes":
        recs.append(RECOMMENDATION_LIBRARY["occupational_chemical_exposure"])
    if raw.get("smoking") == "Yes":
        recs.append(RECOMMENDATION_LIBRARY["smoking"])
    if raw.get("physical_activity") == "Low":
        recs.append(RECOMMENDATION_LIBRARY["physical_activity"])
    if raw.get("sleep_hours", 8) < 6:
        recs.append(RECOMMENDATION_LIBRARY["sleep_hours"])
    if raw.get("stress_level") == "High":
        recs.append(RECOMMENDATION_LIBRARY["stress_level"])
    if raw.get("fruit_vegetable_frequency_1_5", 5) <= 2:
        recs.append(RECOMMENDATION_LIBRARY["fruit_vegetable_frequency_1_5"])
    if raw.get("processed_food_frequency_1_5", 1) >= 4:
        recs.append(RECOMMENDATION_LIBRARY["processed_food_frequency_1_5"])
    return recs[:4]


def predict_risk(raw: dict, bundle: ModelBundle) -> dict:
    """Run the full KRM prediction + SHAP explanation + recommendations
    for one respondent. Returns a plain dict, ready for both the
    template and the Excel record."""
    X_row = engineer_single_respondent(raw, bundle)

    pred_idx = int(bundle.krm_model.predict(X_row.values)[0])
    proba = bundle.krm_model.predict_proba(X_row.values)[0]

    shap_vals = bundle.explainer(X_row)
    contributions = shap_vals[0, :, pred_idx].values
    order = np.argsort(-np.abs(contributions))
    # NOTE: a positive SHAP value means this feature pushed TOWARD the
    # predicted class (supports it); negative means it pulled AWAY from
    # it. This is *not* the same as "increases/decreases risk" in a
    # fixed sense -- that mapping only holds when the predicted class is
    # High. For a Low prediction, a positive contribution is protective
    # (lower risk), so labelling it "increases risk" would be backwards.
    # Framing relative to the predicted classification is correct no
    # matter which of the three classes was predicted.
    top_factors = [
        {
            "feature": bundle.feature_cols[i],
            "supports_prediction": bool(contributions[i] > 0),
            "value": round(float(contributions[i]), 3),
        }
        for i in order[:5]
    ]

    return {
        "predicted_class": CLASS_NAMES[pred_idx],
        "predicted_class_idx": pred_idx,
        "probabilities": {c: float(p) for c, p in zip(CLASS_NAMES, proba)},
        "top_factors": top_factors,
        "recommendations": generate_recommendations(raw),
    }
