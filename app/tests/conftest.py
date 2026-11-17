import sys, io
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sqlite_store
import model_service

HIGH = dict(full_name="Test High", age=31, province="Harare", residence="Urban", bmi=31.2,
    processed_food_frequency_1_5=4, fruit_vegetable_frequency_1_5=2, legume_frequency_1_5=3,
    alcohol_use="Occasional", physical_activity="Low", sleep_hours=5.4, stress_level="High", smoking="No",
    pesticide_exposure="None", air_pollution_exposure="High", occupational_chemical_exposure="No",
    irregular_menstrual_cycle="Yes", severe_menstrual_pain="Yes",
    family_reproductive_history="Yes", previous_reproductive_condition="No")
LOW = dict(full_name="Test Low", age=26, province="Bulawayo", residence="Urban", bmi=22.0,
    processed_food_frequency_1_5=2, fruit_vegetable_frequency_1_5=5, legume_frequency_1_5=4,
    alcohol_use="None", physical_activity="High", sleep_hours=8.0, stress_level="Low", smoking="No",
    pesticide_exposure="None", air_pollution_exposure="Low", occupational_chemical_exposure="No",
    irregular_menstrual_cycle="No", severe_menstrual_pain="No",
    family_reproductive_history="No", previous_reproductive_condition="No")

@pytest.fixture(scope="session")
def bundle():
    return model_service.load_model_bundle()

@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Flask test client wired to a throw-away SQLite database so tests never touch real data."""
    monkeypatch.setattr(sqlite_store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(sqlite_store, "DB_PATH", tmp_path / "test.db")
    sqlite_store.ensure_store_exists()
    import app as app_module
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()

@pytest.fixture()
def high(): return dict(HIGH)
@pytest.fixture()
def low(): return dict(LOW)
