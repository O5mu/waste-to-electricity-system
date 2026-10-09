"""The 60-minute power forecast. Run with:  pytest"""

import json
import random

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from wte import database
from wte.config import FORECAST_METRICS_PATH, FORECAST_MODEL_PATH
from wte.forecast import WINDOW_MIN, forecast, to_minutes
from wte.plant import PlantProcess
from wte.simulator import backfill


class ConstantModel:
    def predict(self, rows):
        assert not rows.isna().any(axis=None)
        return [6.0] * len(rows)


def simulate(minutes: float, start="2026-05-04 08:00:00", interval=2.0):
    plant = PlantProcess(random.Random(1), fault_rate=0.05)
    start = pd.Timestamp(start)
    feeds = [dict(f, timestamp=start) for f in plant.pop_feeds()]
    readings = []
    for step in range(1, int(minutes * 60 / interval) + 1):
        timestamp = start + pd.Timedelta(seconds=step * interval)
        readings.append(dict(plant.read(interval), timestamp=timestamp))
        feeds += [dict(f, timestamp=timestamp) for f in plant.pop_feeds()]
    for feed in feeds:
        for batch in ("next", "then"):
            feed[f"{batch}_feed_at"] = feed["timestamp"] + pd.Timedelta(minutes=feed[f"{batch}_feed_in_min"])
    return readings, feeds


def test_plant_stays_in_the_estimators_range():
    readings, feeds = simulate(8 * 60)
    df = pd.DataFrame(readings)
    assert df["temperature"].between(93, 119).all()
    assert (df["pressure"] < 60).mean() > 0.9
    assert len(feeds) >= 6  # a batch roughly every hour, plus the first charge


def test_a_gap_in_readings_starts_a_new_run():
    readings, _ = simulate(60)
    gap = pd.Timedelta(minutes=10)
    later = [dict(r, timestamp=r["timestamp"] + gap) for r in readings[900:]]
    minutes = to_minutes(pd.DataFrame(readings[:900] + later))
    assert minutes.index[0] > readings[899]["timestamp"]


def test_forecast_needs_history_and_a_feed_log():
    readings, feeds = simulate(WINDOW_MIN - 5)
    assert forecast(ConstantModel(), readings, feeds)["available"] is False

    readings, feeds = simulate(WINDOW_MIN + 10)
    result = forecast(ConstantModel(), readings, [])
    assert result["available"] is False and "feed" in result["reason"]

    result = forecast(ConstantModel(), readings, feeds)
    assert result["available"] and result["predicted_power_w"] == 6.0


@pytest.mark.skipif(not FORECAST_MODEL_PATH.exists(), reason="forecast model not trained")
def test_forecast_endpoint_after_backfill(tmp_path, monkeypatch):
    from wte.api import app

    monkeypatch.setattr(database, "DB_PATH", tmp_path / "forecast.db")
    backfill(PlantProcess(random.Random(7)), minutes=60, interval=2.0)
    with TestClient(app) as client:
        body = client.get("/forecast").json()
        assert body["available"], body
        assert 5.0 < body["predicted_power_w"] < 7.0
        assert database.verify_chain()["intact"]

        feed = {"mass_kg": 0.8, "next_feed_in_min": 55, "next_mass_kg": 0.7,
                "then_feed_in_min": 115, "then_mass_kg": 0.75}
        assert client.post("/feed", json=feed).status_code == 201


@pytest.mark.skipif(not FORECAST_METRICS_PATH.exists(), reason="forecast model not trained")
def test_forecast_beats_the_no_forecast_baseline():
    metrics = json.loads(FORECAST_METRICS_PATH.read_text())
    assert metrics["mae"] < metrics["persistence_mae"]
    assert metrics["r2_score"] > metrics["persistence_r2"]
