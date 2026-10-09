"""SQLite storage for sensor readings.

The log is tamper-evident:

* Triggers reject every UPDATE and DELETE, so the table is append-only through SQLite.
* Each row stores a SHA-256 hash of its own values chained to the previous row's hash.
  Editing or removing a row by other means (e.g. dropping the triggers or patching the
  file) breaks the chain, and `verify_chain()` reports the first row that no longer matches.

Removing only the newest rows leaves a valid, shorter chain; to detect that, keep a copy of
the latest hash elsewhere (`verify_chain()` returns it as `head_hash`).
"""

import hashlib
import sqlite3
from contextlib import contextmanager
from datetime import datetime

from wte.config import DB_PATH

GENESIS_HASH = "0" * 64


@contextmanager
def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def row_hash(prev_hash: str, timestamp: str, temperature: float, pressure: float, voltage: float) -> str:
    """SHA-256 over the previous hash and this row's values."""
    payload = f"{prev_hash}|{timestamp}|{temperature!r}|{pressure!r}|{voltage!r}"
    return hashlib.sha256(payload.encode()).hexdigest()


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
                voltage     REAL NOT NULL,
                prev_hash   TEXT,
                hash        TEXT
            )
            """
        )
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(sensor_readings)")}
        if "hash" not in columns:
            _upgrade_unchained_table(conn)

        for action in ("UPDATE", "DELETE"):
            conn.execute(
                f"""
                CREATE TRIGGER IF NOT EXISTS sensor_readings_no_{action.lower()}
                BEFORE {action} ON sensor_readings
                BEGIN
                    SELECT RAISE(ABORT, 'sensor_readings is append-only');
                END
                """
            )


def _upgrade_unchained_table(conn: sqlite3.Connection) -> None:
    """Add the hash columns to a database created before chaining, and chain its rows."""
    conn.execute("ALTER TABLE sensor_readings ADD COLUMN prev_hash TEXT")
    conn.execute("ALTER TABLE sensor_readings ADD COLUMN hash TEXT")
    prev = GENESIS_HASH
    rows = conn.execute(
        "SELECT id, timestamp, temperature, pressure, voltage FROM sensor_readings ORDER BY id"
    ).fetchall()
    for row in rows:
        current = row_hash(prev, row["timestamp"], row["temperature"], row["pressure"], row["voltage"])
        conn.execute("UPDATE sensor_readings SET prev_hash = ?, hash = ? WHERE id = ?", (prev, current, row["id"]))
        prev = current


def insert_reading(temperature: float, pressure: float, voltage: float) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with connect() as conn:
        # Lock the database so two writers can't both chain onto the same previous row.
        conn.execute("BEGIN IMMEDIATE")
        last = conn.execute("SELECT hash FROM sensor_readings ORDER BY id DESC LIMIT 1").fetchone()
        prev = last["hash"] if last else GENESIS_HASH
        conn.execute(
            "INSERT INTO sensor_readings (timestamp, temperature, pressure, voltage, prev_hash, hash) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (timestamp, temperature, pressure, voltage, prev,
             row_hash(prev, timestamp, temperature, pressure, voltage)),
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


def verify_chain() -> dict:
    """Recompute every hash in order; report whether the log is intact."""
    prev = GENESIS_HASH
    checked = 0
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, timestamp, temperature, pressure, voltage, prev_hash, hash FROM sensor_readings ORDER BY id"
        )
        for row in rows:
            expected = row_hash(prev, row["timestamp"], row["temperature"], row["pressure"], row["voltage"])
            if row["prev_hash"] != prev or row["hash"] != expected:
                return {"intact": False, "rows_checked": checked, "first_bad_id": row["id"], "head_hash": prev}
            prev = row["hash"]
            checked += 1
    return {"intact": True, "rows_checked": checked, "first_bad_id": None, "head_hash": prev}
