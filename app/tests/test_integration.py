"""Integration tests: HTTP routes + model + database working together (Flask test client)."""
import io
import pandas as pd
import sqlite_store as store

def _post(client, payload):
    return client.post("/predict", data=payload)

def test_I01_root_redirects_to_dashboard(client):
    r = client.get("/")
    assert r.status_code == 302 and r.headers["Location"].endswith("/dashboard")

def test_I02_all_pages_load(client):
    for url in ("/dashboard", "/predict", "/members", "/explore"):
        assert client.get(url).status_code == 200, url

def test_I03_predict_returns_result_and_saves(client, high):
    r = _post(client, high)
    assert r.status_code == 200
    assert b"HIGH RISK" in r.data and b"Risk Factor Breakdown" in r.data
    assert len(store.load_members()) == 1

def test_I04_saved_member_appears_in_members_and_dashboard(client, high):
    _post(client, high)
    assert b"Test High" in client.get("/members").data
    assert b"Test High" in client.get("/dashboard").data

def test_I05_members_filter_by_risk(client, high, low):
    _post(client, high); _post(client, low)
    r = client.get("/members?risk=High")
    assert b"Test High" in r.data and b"Test Low" not in r.data

def test_I06_members_search_and_age_filters(client, high, low):
    _post(client, high); _post(client, low)
    assert b"Test Low" in client.get("/members?search=low").data
    r = client.get("/members?min_age=30")
    assert b"Test High" in r.data and b"Test Low" not in r.data

def test_I07_excel_export_downloads_valid_workbook(client, high):
    _post(client, high)
    r = client.get("/members.xlsx")
    assert r.status_code == 200 and "spreadsheetml" in r.headers["Content-Type"]
    df = pd.read_excel(io.BytesIO(r.data))
    assert len(df) == 1 and df.loc[0, "predicted_risk"] == "High"

def test_I08_explore_csv_upload_summary(client):
    csv = b"a,b,c\n1,x,3.5\n2,y,4.5\n3,x,\n"
    r = client.post("/explore", data={"data_file": (io.BytesIO(csv), "t.csv")}, content_type="multipart/form-data")
    assert r.status_code == 200 and b"t.csv" in r.data and b"Missing" in r.data

def test_I09_explore_rejects_bad_extension(client):
    r = client.post("/explore", data={"data_file": (io.BytesIO(b"x"), "evil.exe")}, content_type="multipart/form-data")
    assert b"Unsupported file type" in r.data

def test_I10_explore_requires_a_file(client):
    r = client.post("/explore", data={}, content_type="multipart/form-data")
    assert b"Please choose" in r.data

def test_I11_explore_corrupt_excel_handled(client):
    r = client.post("/explore", data={"data_file": (io.BytesIO(b"not excel"), "bad.xlsx")}, content_type="multipart/form-data")
    assert r.status_code == 200 and b"Could not read this file" in r.data

def test_I12_non_numeric_age_rejected_and_not_saved(client, high):
    _post(client, {**high, "age": "abc"})
    assert len(store.load_members()) == 0

def test_I13_unknown_category_rejected_and_not_saved(client, high):
    r = _post(client, {**high, "physical_activity": "Medium"})
    assert r.status_code == 200 and b"Could not process the form" in r.data
    assert len(store.load_members()) == 0

def test_I14_missing_field_rejected(client, high):
    p = dict(high); p.pop("province")
    _post(client, p)
    assert len(store.load_members()) == 0

def test_I15_xss_in_name_is_escaped(client, high):
    _post(client, {**high, "full_name": "<script>alert(1)</script>"})
    page = client.get("/members").data
    assert b"<script>alert(1)</script>" not in page and b"&lt;script&gt;" in page

def test_I16_sql_injection_in_search_is_harmless(client, high):
    _post(client, high)
    r = client.get("/members?search=' OR '1'='1")
    assert r.status_code == 200 and b"Test High" not in r.data
    assert len(store.load_members()) == 1

def test_I17_oversized_upload_rejected(client):
    big = io.BytesIO(b"a" * (26 * 1024 * 1024))
    r = client.post("/explore", data={"data_file": (big, "big.csv")}, content_type="multipart/form-data")
    assert r.status_code == 413

def test_I18_explore_does_not_chart_identifier_column(client):
    """Regression: the auto-chart must skip high-cardinality columns such as respondent IDs."""
    rows = "\n".join(f"ID{i},{'Urban' if i % 2 else 'Rural'},{i}" for i in range(60))
    csv = ("respondent_id,residence,score\n" + rows).encode()
    r = client.post("/explore", data={"data_file": (io.BytesIO(csv), "ids.csv")}, content_type="multipart/form-data")
    html = r.data.decode()
    assert '<canvas id="exploreChart"' in html          # a chart is still drawn...
    assert "<h2>residence</h2>" in html                 # ...for the genuinely categorical column
    assert "<h2>respondent_id</h2>" not in html         # ...never for the identifier column


def test_I19_out_of_range_numbers_rejected_server_side(client, high):
    """Regression: the HTML min/max limits are bypassable, so the server must enforce them."""
    for field, bad in [("age", "200"), ("age", "17"), ("bmi", "-5"), ("bmi", "nan"), ("sleep_hours", "99"),
                       ("processed_food_frequency_1_5", "9")]:
        r = _post(client, {**high, field: bad})
        assert r.status_code == 200 and b"Could not process the form" in r.data, (field, bad)
    assert len(store.load_members()) == 0

def test_I20_boundary_values_accepted(client, high):
    for field, ok in [("age", "18"), ("age", "45"), ("bmi", "12"), ("bmi", "55"), ("sleep_hours", "3"), ("sleep_hours", "12")]:
        r = _post(client, {**high, field: ok})
        assert b"Could not process the form" not in r.data, (field, ok)
    assert len(store.load_members()) == 6
