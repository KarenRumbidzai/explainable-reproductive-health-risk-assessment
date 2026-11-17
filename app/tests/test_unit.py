"""Unit tests: smallest testable pieces (model_service functions and the SQLite store)."""
import json
import numpy as np
import pytest
import model_service as ms
import sqlite_store as store

# ---------- model_service ----------
def test_U01_bmi_category_boundaries():
    assert ms.bmi_category(18.4) == "Underweight"
    assert ms.bmi_category(18.5) == "Normal"
    assert ms.bmi_category(24.9) == "Normal"
    assert ms.bmi_category(25.0) == "Overweight"
    assert ms.bmi_category(30.0) == "Obese"

def test_U02_model_bundle_loads_all_artifacts(bundle):
    assert len(bundle.feature_cols) == 30
    assert len(bundle.provinces) == 10
    assert bundle.krm_model is not None and bundle.xgb_model is not None

def test_U03_form_options_come_from_training_maps(bundle):
    opts = bundle.form_options
    assert opts["pesticide_exposure"] == ["None", "Low", "Moderate", "High"]
    assert opts["residence"] == ["Urban", "Rural"]
    assert "Bulawayo" in opts["province"]

def test_U04_feature_vector_shape_and_order(bundle, high):
    row = ms.engineer_single_respondent(high, bundle)
    assert row.shape == (1, 30)
    assert list(row.columns) == bundle.feature_cols
    assert row.isna().sum().sum() == 0

def test_U05_encoding_values_correct(bundle, high):
    row = ms.engineer_single_respondent(high, bundle).iloc[0]
    assert row["pesticide_exposure"] == 0          # "None" -> 0
    assert row["air_pollution_exposure"] == 2      # "High" -> 2
    assert row["irregular_menstrual_cycle"] == 1
    assert row["bmi_cat_Obese"] == 1 and row["bmi_cat_Underweight"] == 0
    assert row["province_Harare"] == 1

def test_U06_reference_province_all_zero(bundle, low):
    row = ms.engineer_single_respondent(low, bundle).iloc[0]   # Bulawayo = reference level
    prov_cols = [c for c in bundle.feature_cols if c.startswith("province_")]
    assert len(prov_cols) == 9 and row[prov_cols].sum() == 0

def test_U07_probabilities_sum_to_one(bundle, high):
    r = ms.predict_risk(high, bundle)
    assert abs(sum(r["probabilities"].values()) - 1.0) < 1e-6

def test_U08_high_profile_predicted_high(bundle, high):
    assert ms.predict_risk(high, bundle)["predicted_class"] == "High"

def test_U09_low_profile_predicted_low(bundle, low):
    assert ms.predict_risk(low, bundle)["predicted_class"] == "Low"

def test_U10_prediction_is_deterministic(bundle, high):
    a, b = ms.predict_risk(high, bundle), ms.predict_risk(high, bundle)
    assert a["probabilities"] == b["probabilities"]

def test_U11_shap_returns_five_ranked_factors(bundle, high):
    f = ms.predict_risk(high, bundle)["top_factors"]
    assert len(f) == 5
    assert all(abs(f[i]["value"]) >= abs(f[i+1]["value"]) for i in range(4))
    assert f[0]["feature"] == "irregular_menstrual_cycle"

def test_U12_recommendations_triggered_and_capped(high):
    recs = ms.generate_recommendations({**high, "smoking": "Yes", "occupational_chemical_exposure": "Yes",
                                        "pesticide_exposure": "High"})
    assert 1 <= len(recs) <= 4

def test_U13_no_recommendations_for_clean_profile(low):
    assert ms.generate_recommendations(low) == []

def test_U14_unknown_category_raises(bundle, low):
    with pytest.raises(KeyError):
        ms.engineer_single_respondent({**low, "physical_activity": "Medium"}, bundle)

# ---------- sqlite_store ----------
def _fake_result():
    return {"predicted_class": "High", "probabilities": {"Low": .1, "Moderate": .2, "High": .7},
            "top_factors": [{"feature": "bmi", "supports_prediction": True, "value": 0.5}],
            "recommendations": ["x"]}

def test_U15_store_roundtrip(client, high):
    rec = store.append_member(high, _fake_result())
    df = store.load_members()
    assert len(df) == 1 and rec["id"] == 1
    assert df.loc[0, "predicted_risk"] == "High"
    assert json.loads(df.loc[0, "top_factors_json"])[0]["feature"] == "bmi"

def test_U16_store_ids_autoincrement(client, high):
    a = store.append_member(high, _fake_result()); b = store.append_member(high, _fake_result())
    assert b["id"] == a["id"] + 1

def test_U17_filters(client, high, low):
    store.append_member(high, _fake_result())
    r = _fake_result(); r["predicted_class"] = "Low"
    store.append_member(low, r)
    df = store.load_members()
    assert len(store.filter_members(df, risk="High")) == 1
    assert len(store.filter_members(df, province="Bulawayo")) == 1
    assert len(store.filter_members(df, min_age=30)) == 1
    assert len(store.filter_members(df, search="test")) == 2

def test_U18_sql_injection_stored_as_plain_text(client, high):
    evil = "x'); DROP TABLE members;--"
    store.append_member({**high, "full_name": evil}, _fake_result())
    df = store.load_members()                    # table must still exist
    assert df.loc[0, "full_name"] == evil
