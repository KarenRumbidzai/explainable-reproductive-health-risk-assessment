"""System tests: whole-application scenarios end to end, plus non-functional checks."""
import time, statistics
import numpy as np
import pandas as pd
import pytest
import model_service as ms
import sqlite_store as store
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "reproductive_health_dataset.csv"

def test_S01_end_to_end_register_review_export(client, high, low):
    """Register two members, see them on dashboard + members page, filter, export."""
    client.post("/predict", data=high); client.post("/predict", data=low)
    dash = client.get("/dashboard").data
    assert b"Members Registered" in dash
    assert len(store.load_members()) == 2
    assert client.get("/members.xlsx").status_code == 200

def test_S02_risk_ordering_is_clinically_sensible(bundle, low, high):
    """Adding risk factors to a low-risk profile must not reduce P(High)."""
    base = ms.predict_risk(low, bundle)["probabilities"]["High"]
    worse = ms.predict_risk({**low, "irregular_menstrual_cycle": "Yes", "previous_reproductive_condition": "Yes"}, bundle)["probabilities"]["High"]
    assert worse > base

def test_S03_every_form_option_accepted_by_model(bundle, low):
    """Each dropdown choice the UI offers must produce a valid prediction (no crash)."""
    for field, values in bundle.form_options.items():
        for v in values:
            r = ms.predict_risk({**low, field: v}, bundle)
            assert r["predicted_class"] in ("Low", "Moderate", "High")

def test_S04_model_matches_saved_manifest_accuracy(bundle):
    """Re-score the app's own copy of the dataset: held-out accuracy recorded in the manifest must be reproducible."""
    import json
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(DATA, keep_default_na=False)
    X = pd.DataFrame([ms.engineer_single_respondent(r.to_dict(), bundle).iloc[0] for _, r in df.iterrows()])[bundle.feature_cols]
    y = df["reproductive_health_risk"].map({"Low": 0, "Moderate": 1, "High": 2}).astype(int)
    _, Xte, _, yte = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    acc = (bundle.krm_model.predict(Xte.values) == yte.values).mean()
    manifest = json.load(open(Path(DATA).parent.parent / "models" / "model_manifest.json"))
    # the manifest stores metrics rounded to 4 decimal places
    assert round(acc, 4) == manifest["test_metrics_krm"]["Accuracy"]

def test_S05_prediction_latency(bundle, high):
    ms.predict_risk(high, bundle)                      # warm-up
    times = []
    for _ in range(30):
        t = time.perf_counter(); ms.predict_risk(high, bundle); times.append(time.perf_counter() - t)
    print(f"\nPREDICTION LATENCY mean={statistics.mean(times)*1000:.1f}ms p95={np.percentile(times,95)*1000:.1f}ms max={max(times)*1000:.1f}ms")
    assert statistics.mean(times) < 2.0

def test_S06_http_round_trip_latency(client, high):
    client.post("/predict", data=high)                 # warm-up
    times = []
    for _ in range(15):
        t = time.perf_counter(); r = client.post("/predict", data=high); times.append(time.perf_counter() - t)
        assert r.status_code == 200
    print(f"\nHTTP /predict round trip mean={statistics.mean(times)*1000:.1f}ms max={max(times)*1000:.1f}ms")
    assert statistics.mean(times) < 3.0

def test_S07_persistence_across_connections(client, high):
    client.post("/predict", data=high)
    first = store.load_members()
    again = store.load_members()                       # new connection each call
    assert first.equals(again) and len(again) == 1

def test_S08_dashboard_handles_empty_and_populated_store(client, high):
    assert client.get("/dashboard").status_code == 200   # empty
    client.post("/predict", data=high)
    assert client.get("/dashboard").status_code == 200   # populated

def test_S09_batch_of_50_registrations(client, high):
    for i in range(50):
        r = client.post("/predict", data={**high, "full_name": f"Batch {i}", "age": 20 + i % 25})
        assert r.status_code == 200
    assert len(store.load_members()) == 50
    assert b"50" in client.get("/dashboard").data
