"""SQLite storage for sensor readings."""

import sqlite3
from contextlib import contextmanager
from datetime import datetime

from wte.config import DB_PATH


@contextmanager
def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sensor_readings (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT NOT NULL,
                temperature REAL NOT NULL,
                pressure    REAL NOT NULL,
                voltage     REAL NOT NULL
            )
            """
        )


def insert_reading(temperature: float, pressure: float, voltage: float) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with connect() as conn:
        conn.execute(
            "INSERT INTO sensor_readings (timestamp, temperature, pressure, voltage) VALUES (?, ?, ?, ?)",
            (timestamp, temperature, pressure, voltage),
        )
    return timestamp


def latest_readings(limit: int) -> list[dict]:
    """Most recent readings, newest first."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT timestamp, temperature, pressure, voltage FROM sensor_readings ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]
