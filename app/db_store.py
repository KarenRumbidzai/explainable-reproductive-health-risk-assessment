"""
db_store.py

MySQL-backed storage for registered members and their KRM prediction
results -- a drop-in replacement for the earlier Excel-based
data_store.py. Same three functions (load_members, append_member,
filter_members) with the same shapes in and out, so app.py's routes
don't need to change, only the import at the top.

Connection settings live in db_config.py, not here -- edit that file to
point at your MySQL server.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text

import db_config

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

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

_engine = None


def get_engine():
    """A single, reused SQLAlchemy engine (it manages its own small
    connection pool internally, so this is safe to call from every
    request rather than opening a raw connection each time)."""
    global _engine
    if _engine is None:
        url = (
            f"mysql+mysqlconnector://{db_config.DB_USER}:{db_config.DB_PASSWORD}"
            f"@{db_config.DB_HOST}:{db_config.DB_PORT}/{db_config.DB_NAME}"
        )
        _engine = create_engine(url, pool_pre_ping=True)
    return _engine


def ensure_store_exists() -> None:
    """Create the members table if it doesn't exist yet, using schema.sql.
    Safe to call every time the app starts -- CREATE TABLE IF NOT EXISTS
    is a no-op once the table is already there."""
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"schema.sql not found at {SCHEMA_PATH}")

    statement = SCHEMA_PATH.read_text()
    engine = get_engine()
    with engine.begin() as conn:
        for chunk in statement.split(";"):
            chunk = chunk.strip()
            if chunk:
                conn.execute(text(chunk))


def load_members() -> pd.DataFrame:
    engine = get_engine()
    df = pd.read_sql(f"SELECT {', '.join(COLUMNS)} FROM members ORDER BY id", engine)
    # MySQL returns DECIMAL columns as Decimal objects; convert to plain
    # floats so the rest of the app (Jinja rounding, Excel export) works
    # the same way it did against the Excel-backed version.
    for col in ["sleep_hours", "bmi", "prob_low", "prob_moderate", "prob_high"]:
        df[col] = df[col].astype(float)
    return df


def append_member(raw: dict, result: dict) -> dict:
    """Insert one new member + their prediction result as a new row.
    Returns the full record, including the auto-generated id."""
    record = {
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
    }

    columns = list(record.keys())
    placeholders = ", ".join(f":{c}" for c in columns)
    sql = text(f"INSERT INTO members ({', '.join(columns)}) VALUES ({placeholders})")

    engine = get_engine()
    with engine.begin() as conn:
        result_proxy = conn.execute(sql, record)
        new_id = result_proxy.lastrowid

    record["id"] = new_id
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
    """Same signature as the Excel-backed version's filter_members, so
    app.py's /members route needs no changes. Filtering happens in
    pandas on the already-loaded DataFrame -- fine at this scale; for a
    much larger table, push these conditions into the SQL WHERE clause
    in load_members() instead."""
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


def export_to_excel(path: Path) -> Path:
    """Generate an Excel workbook from the current MySQL contents, on
    demand -- used by the /members.xlsx download route. MySQL is the
    source of truth; this file is a snapshot, not something the app
    reads back from."""
    df = load_members()
    df.to_excel(path, sheet_name="Members", index=False)
    return path
