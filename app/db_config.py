"""
db_config.py

Edit these four values to match your MySQL setup, then leave this file
alone -- nothing else in the app needs to change.

If you'd rather not put a password in a plain file, you can instead set
these as environment variables (DB_HOST, DB_USER, DB_PASSWORD, DB_NAME)
before running the app -- any of them left unset here falls back to the
environment variable, then to the default shown.
"""

import os

DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = int(os.environ.get("DB_PORT", "3306"))
DB_USER = os.environ.get("DB_USER", "krm_app")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "krm_app_password")
DB_NAME = os.environ.get("DB_NAME", "krm_db")
