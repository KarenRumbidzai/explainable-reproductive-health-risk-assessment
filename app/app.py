"""
app.py

KRM Reproductive Health Risk Assessment -- Flask web application.

Routes:
    /               redirects to /dashboard
    /dashboard      overview: training-set context + live registered-member stats
    /predict        register a new member, get a KRM prediction + explanation
    /members        browse registered members, with filters
    /members.xlsx   download the underlying Excel workbook directly
    /explore        upload any CSV/Excel file for a quick exploratory summary

Model, feature-engineering, and prediction logic live in model_service.py.
Member storage defaults to SQLite (sqlite_store.py, imported here as
data_store) -- zero setup, one file, right for single-user local use.
MySQL (db_store.py) and Excel (data_store.py) are also included as
drop-in alternatives with the same three functions; see README.md to
switch.
Run:  python app.py   then open http://127.0.0.1:5050
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from flask import (
    Flask, render_template, request, redirect, url_for, flash, send_file
)

import sqlite_store as data_store
import model_service

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "reproductive_health_dataset.csv"
ALLOWED_UPLOAD_EXTENSIONS = {"csv", "xlsx", "xls"}

# Server-side numeric limits. These mirror the min/max attributes on the HTML form,
# but are enforced here too because browser-side limits can be bypassed by a crafted request.
NUMERIC_LIMITS = {
    "age": (18, 45),
    "bmi": (12.0, 55.0),
    "sleep_hours": (3.0, 12.0),
    "processed_food_frequency_1_5": (1, 5),
    "fruit_vegetable_frequency_1_5": (1, 5),
    "legume_frequency_1_5": (1, 5),
}


def validate_numeric_ranges(raw: dict) -> None:
    """Raise ValueError if any numeric answer is non-finite or outside its allowed range."""
    for field, (lo, hi) in NUMERIC_LIMITS.items():
        value = raw[field]
        if value != value or value in (float("inf"), float("-inf")) or not (lo <= value <= hi):
            raise ValueError(f"{field.replace('_', ' ')} must be between {lo} and {hi}")


app = Flask(__name__)
app.config["SECRET_KEY"] = "krm-dev-secret-change-in-production"
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB upload cap

# Loaded once at startup; reused across every request.
MODEL_BUNDLE = model_service.load_model_bundle()


# =======================================================================
# Routes
# =======================================================================
@app.route("/")
def index():
    return redirect(url_for("dashboard"))


@app.route("/dashboard")
def dashboard():
    dataset_stats = None
    if DATASET_PATH.exists():
        df = pd.read_csv(DATASET_PATH, keep_default_na=False, na_values=[""])
        dist = df["reproductive_health_risk"].value_counts()
        dataset_stats = {
            "n_respondents": len(df),
            "low_pct": round(dist.get("Low", 0) / len(df) * 100, 1),
            "moderate_pct": round(dist.get("Moderate", 0) / len(df) * 100, 1),
            "high_pct": round(dist.get("High", 0) / len(df) * 100, 1),
        }

    members = data_store.load_members()
    member_count = len(members)

    risk_dist = {"Low": 0, "Moderate": 0, "High": 0}
    if member_count:
        risk_dist.update(members["predicted_risk"].value_counts().to_dict())

    top_factor_counts = {}
    if member_count:
        top_factor_counts = (
            members["top_factor"].value_counts().head(6).to_dict()
        )

    # Registrations per day, for a small trend chart -- naturally sparse
    # early on, but grows meaningful as more members are registered.
    trend = {"labels": [], "counts": []}
    if member_count:
        dates = pd.to_datetime(members["registered_at"]).dt.date
        counts = dates.value_counts().sort_index()
        trend = {"labels": [str(d) for d in counts.index], "counts": counts.values.tolist()}

    recent_members = members.sort_values("id", ascending=False).head(6).to_dict("records")

    return render_template(
        "dashboard.html",
        dataset_stats=dataset_stats,
        member_count=member_count,
        risk_dist=risk_dist,
        top_factor_counts=top_factor_counts,
        trend=trend,
        recent_members=recent_members,
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    submitted: dict = {}

    if request.method == "POST":
        try:
            raw = {
                "full_name": request.form["full_name"].strip() or "Unnamed",
                "age": int(request.form["age"]),
                "province": request.form["province"],
                "residence": request.form["residence"],
                "processed_food_frequency_1_5": int(request.form["processed_food_frequency_1_5"]),
                "fruit_vegetable_frequency_1_5": int(request.form["fruit_vegetable_frequency_1_5"]),
                "legume_frequency_1_5": int(request.form["legume_frequency_1_5"]),
                "physical_activity": request.form["physical_activity"],
                "sleep_hours": float(request.form["sleep_hours"]),
                "stress_level": request.form["stress_level"],
                "bmi": float(request.form["bmi"]),
                "pesticide_exposure": request.form["pesticide_exposure"],
                "air_pollution_exposure": request.form["air_pollution_exposure"],
                "occupational_chemical_exposure": request.form["occupational_chemical_exposure"],
                "smoking": request.form["smoking"],
                "alcohol_use": request.form["alcohol_use"],
                "irregular_menstrual_cycle": request.form["irregular_menstrual_cycle"],
                "severe_menstrual_pain": request.form["severe_menstrual_pain"],
                "family_reproductive_history": request.form["family_reproductive_history"],
                "previous_reproductive_condition": request.form["previous_reproductive_condition"],
            }
            submitted = raw
            validate_numeric_ranges(raw)

            result = model_service.predict_risk(raw, MODEL_BUNDLE)
            data_store.append_member(raw, result)

        except (KeyError, ValueError) as exc:
            flash(f"Could not process the form: {exc}", "error")

    return render_template(
        "predict.html",
        options=MODEL_BUNDLE.form_options,
        result=result,
        submitted=submitted,
        disclaimer=model_service.DISCLAIMER,
    )


@app.route("/members")
def members():
    df = data_store.load_members()

    filters = {
        "risk": request.args.get("risk", "").strip(),
        "province": request.args.get("province", "").strip(),
        "residence": request.args.get("residence", "").strip(),
        "min_age": request.args.get("min_age", "").strip(),
        "max_age": request.args.get("max_age", "").strip(),
        "search": request.args.get("search", "").strip(),
    }
    min_age = int(filters["min_age"]) if filters["min_age"].isdigit() else None
    max_age = int(filters["max_age"]) if filters["max_age"].isdigit() else None

    filtered = data_store.filter_members(
        df,
        risk=filters["risk"] or None,
        province=filters["province"] or None,
        residence=filters["residence"] or None,
        min_age=min_age,
        max_age=max_age,
        search=filters["search"] or None,
    ).sort_values("id", ascending=False)

    provinces = sorted(df["province"].dropna().unique().tolist()) if len(df) else []

    return render_template(
        "members.html",
        members=filtered.to_dict("records"),
        total_count=len(df),
        filtered_count=len(filtered),
        filters=filters,
        provinces=provinces,
    )


@app.route("/members.xlsx")
def download_members():
    export_path = BASE_DIR / "data" / "members_export.xlsx"
    export_path.parent.mkdir(exist_ok=True)
    data_store.export_to_excel(export_path)
    return send_file(
        export_path,
        as_attachment=True,
        download_name="krm_members.xlsx",
    )


@app.route("/explore", methods=["GET", "POST"])
def explore():
    summary = None
    error = None

    if request.method == "POST":
        file = request.files.get("data_file")
        if not file or file.filename == "":
            error = "Please choose a CSV or Excel file to upload."
        else:
            ext = file.filename.rsplit(".", 1)[-1].lower()
            if ext not in ALLOWED_UPLOAD_EXTENSIONS:
                error = "Unsupported file type. Please upload .csv, .xlsx, or .xls."
            else:
                try:
                    if ext == "csv":
                        edf = pd.read_csv(file, keep_default_na=False, na_values=[""])
                    else:
                        edf = pd.read_excel(file)

                    numeric_df = edf.select_dtypes(include=[np.number])
                    describe_html = (
                        numeric_df.describe().T.round(2).to_html(classes="data-table", border=0)
                        if not numeric_df.empty else None
                    )

                    missing = edf.isnull().sum()
                    missing = missing[missing > 0]

                    categorical_summaries = {}
                    chart_col, chart_data = None, None
                    for col in edf.select_dtypes(exclude=[np.number]).columns[:8]:
                        vc = edf[col].value_counts().head(8)
                        categorical_summaries[col] = vc.to_dict()
                        # only chart genuinely categorical columns; skip identifiers/free text
                        if chart_col is None and 1 < edf[col].nunique() <= 12:
                            chart_col = col
                            chart_data = {"labels": vc.index.astype(str).tolist(),
                                           "counts": vc.values.tolist()}

                    summary = {
                        "filename": file.filename,
                        "n_rows": len(edf),
                        "n_cols": edf.shape[1],
                        "dtypes": {c: str(t) for c, t in edf.dtypes.items()},
                        "missing": missing.to_dict(),
                        "describe_html": describe_html,
                        "categorical_summaries": categorical_summaries,
                        "chart_col": chart_col,
                        "chart_data": chart_data,
                        "preview_html": edf.head(10).to_html(classes="data-table", border=0, index=False),
                    }
                except Exception as exc:
                    error = f"Could not read this file: {exc}"

    return render_template("explore.html", summary=summary, error=error)


if __name__ == "__main__":
    data_store.ensure_store_exists()
    app.run(host="127.0.0.1", port=5050, debug=True)
