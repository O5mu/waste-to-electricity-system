"""Tamper-evidence of the sensor log. Run with:  pytest"""

import sqlite3

import pytest
from fastapi.testclient import TestClient

from wte import database


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    monkeypatch.setattr(database, "DB_PATH", path)
    database.init_db()
    for i in range(5):
        database.insert_reading(100 + i * 0.37, 50.1 + i, 13.6)
    return path


def test_new_log_verifies(db):
    result = database.verify_chain()
    assert result["intact"] and result["rows_checked"] == 5


@pytest.mark.parametrize("statement", [
    "UPDATE sensor_readings SET pressure = 1 WHERE id = 3",
    "DELETE FROM sensor_readings WHERE id = 3",
])
def test_edits_and_deletes_are_rejected(db, statement):
    with sqlite3.connect(db) as conn, pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute(statement)


def test_edit_behind_the_triggers_is_detected(db):
    with sqlite3.connect(db) as conn:
        conn.execute("DROP TRIGGER sensor_readings_no_update")
        conn.execute("UPDATE sensor_readings SET pressure = 30.0 WHERE id = 3")
    result = database.verify_chain()
    assert not result["intact"] and result["first_bad_id"] == 3


def test_deleted_row_is_detected(db):
    with sqlite3.connect(db) as conn:
        conn.execute("DROP TRIGGER sensor_readings_no_delete")
        conn.execute("DELETE FROM sensor_readings WHERE id = 2")
    result = database.verify_chain()
    assert not result["intact"] and result["first_bad_id"] == 3


def test_database_from_previous_version_is_upgraded(tmp_path, monkeypatch):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE sensor_readings (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, "
            "temperature REAL NOT NULL, pressure REAL NOT NULL, voltage REAL NOT NULL)"
        )
        conn.executemany(
            "INSERT INTO sensor_readings (timestamp, temperature, pressure, voltage) VALUES (?, ?, ?, ?)",
            [("2026-05-04 10:00:00", 101.2, 55.4, 13.6), ("2026-05-04 10:00:02", 102.9, 57.0, 13.7)],
        )
    monkeypatch.setattr(database, "DB_PATH", path)
    database.init_db()
    database.insert_reading(103.0, 58.0, 13.7)
    result = database.verify_chain()
    assert result["intact"] and result["rows_checked"] == 3


def test_integrity_endpoint(db):
    from wte.api import app

    with TestClient(app) as client:
        client.post("/update-reading", json={"temperature": 101.0, "pressure": 61.0, "voltage": 13.6})
        body = client.get("/integrity").json()
        assert body["intact"] and body["rows_checked"] == 6
        assert client.get("/latest-reading").json()["is_critical"] is True
