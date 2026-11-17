"""
data_store.py

Stores registered members and their KRM prediction results in a plain
Excel workbook (data/members.xlsx) rather than a database -- per
requirement, so the data is directly viewable/shareable in Excel without
any extra tooling.

Every write re-reads the current file, appends the new row, and writes
the whole sheet back out. That's the right trade-off for a single-user
local app with a human-scale number of registrations (hundreds to a few
thousand rows); it would need to become a real database if this were
ever multi-user or high-volume.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
MEMBERS_XLSX = DATA_DIR / "members.xlsx"
SHEET_NAME = "Members"

COLUMNS = [
    "id", "full_name", "age", "province", "residence",
    "processed_food_frequency_1_5", "fruit_vegetable_frequency_1_5",
    "legume_frequency_1_5", "physical_activity", "sleep_hours", "stress_level",
    "bmi", "pesticide_exposure", "air_pollution_exposure",
    "occupational_chemical_exposure", "smoking", "alcohol_use",
    "irregular_menstrual_cycle", "severe_menstrual_pain",
    "family_reproductive_history", "previous_reproductive_condition",
    "predicted_risk", "prob_low", "prob_moderate", "prob_high",
    "top_factor", "top_factors_json", "recommendations_json",
    "registered_at",
]


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def ensure_store_exists() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    if not MEMBERS_XLSX.exists():
        _empty_frame().to_excel(MEMBERS_XLSX, sheet_name=SHEET_NAME, index=False)


def load_members() -> pd.DataFrame:
    ensure_store_exists()
    df = pd.read_excel(MEMBERS_XLSX, sheet_name=SHEET_NAME)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = pd.Series(dtype="object")
    return df[COLUMNS]


def append_member(raw: dict, result: dict) -> dict:
    """Append one new member + their prediction result as a new row in
    the Excel workbook. Returns the full record that was written."""
    df = load_members()
    new_id = int(df["id"].max()) + 1 if len(df) else 1

    record = {
        "id": new_id,
        "full_name": raw["full_name"],
        "age": raw["age"],
        "province": raw["province"],
        "residence": raw["residence"],
        "processed_food_frequency_1_5": raw["processed_food_frequency_1_5"],
        "fruit_vegetable_frequency_1_5": raw["fruit_vegetable_frequency_1_5"],
        "legume_frequency_1_5": raw["legume_frequency_1_5"],
        "physical_activity": raw["physical_activity"],
        "sleep_hours": raw["sleep_hours"],
        "stress_level": raw["stress_level"],
        "bmi": raw["bmi"],
        "pesticide_exposure": raw["pesticide_exposure"],
        "air_pollution_exposure": raw["air_pollution_exposure"],
        "occupational_chemical_exposure": raw["occupational_chemical_exposure"],
        "smoking": raw["smoking"],
        "alcohol_use": raw["alcohol_use"],
        "irregular_menstrual_cycle": raw["irregular_menstrual_cycle"],
        "severe_menstrual_pain": raw["severe_menstrual_pain"],
        "family_reproductive_history": raw["family_reproductive_history"],
        "previous_reproductive_condition": raw["previous_reproductive_condition"],
        "predicted_risk": result["predicted_class"],
        "prob_low": result["probabilities"]["Low"],
        "prob_moderate": result["probabilities"]["Moderate"],
        "prob_high": result["probabilities"]["High"],
        "top_factor": result["top_factors"][0]["feature"] if result["top_factors"] else "",
        "top_factors_json": json.dumps(result["top_factors"]),
        "recommendations_json": json.dumps(result["recommendations"]),
        "registered_at": datetime.now().isoformat(timespec="seconds"),
    }

    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    df.to_excel(MEMBERS_XLSX, sheet_name=SHEET_NAME, index=False)
    return record


def filter_members(
    df: pd.DataFrame,
    risk: str | None = None,
    province: str | None = None,
    residence: str | None = None,
    min_age: int | None = None,
    max_age: int | None = None,
    search: str | None = None,
) -> pd.DataFrame:
    """Apply optional filters to the members table. Every argument is
    optional and ignored when not provided (None or empty string)."""
    out = df
    if risk:
        out = out[out["predicted_risk"] == risk]
    if province:
        out = out[out["province"] == province]
    if residence:
        out = out[out["residence"] == residence]
    if min_age is not None:
        out = out[out["age"] >= min_age]
    if max_age is not None:
        out = out[out["age"] <= max_age]
    if search:
        out = out[out["full_name"].astype(str).str.contains(search, case=False, na=False)]
    return out
