"""
sqlite_store.py

SQLite-backed storage for registered members and their KRM prediction
results -- the "just works, nothing to install" option. Same three
functions (load_members, append_member, filter_members) with the same
shapes in and out as data_store.py (Excel) and db_store.py (MySQL), so
app.py's routes don't need to change, only the import at the top.

Unlike MySQL, there is no server to install, configure, or keep running
-- the whole database is one file (data/krm.db), created automatically
on first run. This is the right choice for a single-user, single-machine
app; reach for MySQL instead only if multiple people or machines need to
read/write the same data at once.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
DB_PATH = DATA_DIR / "krm.db"

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

SCHEMA = """
CREATE TABLE IF NOT EXISTS members (
    id                                  INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name                           TEXT NOT NULL,
    age                                 INTEGER NOT NULL,
    province                            TEXT NOT NULL,
    residence                           TEXT NOT NULL,
    processed_food_frequency_1_5        INTEGER NOT NULL,
    fruit_vegetable_frequency_1_5       INTEGER NOT NULL,
    legume_frequency_1_5                INTEGER NOT NULL,
    physical_activity                   TEXT NOT NULL,
    sleep_hours                         REAL NOT NULL,
    stress_level                        TEXT NOT NULL,
    bmi                                 REAL NOT NULL,
    pesticide_exposure                  TEXT NOT NULL,
    air_pollution_exposure              TEXT NOT NULL,
    occupational_chemical_exposure      TEXT NOT NULL,
    smoking                             TEXT NOT NULL,
    alcohol_use                         TEXT NOT NULL,
    irregular_menstrual_cycle           TEXT NOT NULL,
    severe_menstrual_pain               TEXT NOT NULL,
    family_reproductive_history         TEXT NOT NULL,
    previous_reproductive_condition     TEXT NOT NULL,
    predicted_risk                      TEXT NOT NULL,
    prob_low                            REAL NOT NULL,
    prob_moderate                       REAL NOT NULL,
    prob_high                           REAL NOT NULL,
    top_factor                          TEXT,
    top_factors_json                    TEXT,
    recommendations_json                TEXT,
    registered_at                       TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);
CREATE INDEX IF NOT EXISTS idx_predicted_risk ON members(predicted_risk);
CREATE INDEX IF NOT EXISTS idx_province ON members(province);
CREATE INDEX IF NOT EXISTS idx_registered_at ON members(registered_at);
"""


def get_connection() -> sqlite3.Connection:
    DATA_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_store_exists() -> None:
    """Create the members table (and its indexes) if they don't exist
    yet. Safe to call every time the app starts."""
    conn = get_connection()
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def load_members() -> pd.DataFrame:
    ensure_store_exists()
    conn = get_connection()
    try:
        df = pd.read_sql(f"SELECT {', '.join(COLUMNS)} FROM members ORDER BY id", conn)
    finally:
        conn.close()
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
    placeholders = ", ".join(["?"] * len(columns))
    sql = f"INSERT INTO members ({', '.join(columns)}) VALUES ({placeholders})"

    conn = get_connection()
    try:
        cursor = conn.execute(sql, list(record.values()))
        conn.commit()
        new_id = cursor.lastrowid
    finally:
        conn.close()

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
    """Same signature as the Excel- and MySQL-backed versions, so
    app.py's /members route needs no changes regardless of which store
    is active."""
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
    """Generate an Excel workbook from the current SQLite contents, on
    demand -- used by the /members.xlsx download route."""
    df = load_members()
    df.to_excel(path, sheet_name="Members", index=False)
    return path
