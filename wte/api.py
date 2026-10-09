"""FastAPI backend: stores sensor readings and serves them with ML power estimates.

Run with:  uvicorn wte.api:app --reload
"""

import json
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from wte import __version__, database
from wte.config import (FEATURES, FORECAST_METRICS_PATH, FORECAST_MODEL_PATH, METRICS_PATH, MODEL_PATH,
                        PRESSURE_LIMIT_PSI)
from wte.forecast import RUN_AGE_CAP_MIN, forecast

state: dict = {"model": None, "metrics": None, "forecast_model": None, "forecast_metrics": None}


@asynccontextmanager
async def lifespan(_: FastAPI):
    database.init_db()
    for key, path in (("model", MODEL_PATH), ("forecast_model", FORECAST_MODEL_PATH)):
        if path.exists():
            state[key] = joblib.load(path)
    for key, path in (("metrics", METRICS_PATH), ("forecast_metrics", FORECAST_METRICS_PATH)):
        if path.exists():
            state[key] = json.loads(path.read_text())
    yield


app = FastAPI(
    title="WTE System API",
    description="Backend for the Integrated Waste-to-Electricity monitoring system.",
    version=__version__,
    lifespan=lifespan,
)


class SensorReading(BaseModel):
    temperature: float = Field(..., ge=-50, le=1500, description="Combustion temperature (°C)")
    pressure: float = Field(..., ge=0, le=500, description="Chamber pressure (PSI)")
    voltage: float = Field(..., ge=0, le=100, description="Generator output voltage (V)")


class FeedEvent(BaseModel):
    mass_kg: float = Field(..., gt=0, le=50, description="Weight of the batch loaded now (kg)")
    next_feed_in_min: float = Field(..., gt=0, le=600, description="Minutes until the next planned batch")
    next_mass_kg: float = Field(..., gt=0, le=50, description="Planned weight of the next batch (kg)")
    then_feed_in_min: float = Field(..., gt=0, le=600, description="Minutes until the batch after that")
    then_mass_kg: float = Field(..., gt=0, le=50, description="Planned weight of the batch after that (kg)")


def with_predictions(readings: list[dict]) -> list[dict]:
    """Attach the model's power estimate and the safety flag to each reading."""
    if readings and state["model"] is not None:
        features = pd.DataFrame(
            [[r["temperature"], r["pressure"]] for r in readings], columns=FEATURES
        )
        predictions = state["model"].predict(features)
    else:
        predictions = [None] * len(readings)

    for reading, prediction in zip(readings, predictions):
        reading["predicted_power_w"] = None if prediction is None else round(float(prediction), 4)
        reading["is_critical"] = reading["pressure"] >= PRESSURE_LIMIT_PSI
    return readings


@app.get("/")
def root():
    return {"message": "Integrated Waste-to-Electricity API is running.", "docs": "/docs"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": __version__,
        "model_loaded": state["model"] is not None,
        "forecast_model_loaded": state["forecast_model"] is not None,
        "pressure_limit_psi": PRESSURE_LIMIT_PSI,
    }


@app.post("/update-reading", status_code=201)
def update_reading(reading: SensorReading):
    """Called by the serial bridge (hardware) or the simulator for each new reading."""
    timestamp = database.insert_reading(reading.temperature, reading.pressure, reading.voltage)
    return {"status": "success", "timestamp": timestamp}


@app.get("/latest-reading")
def latest_reading():
    readings = database.latest_readings(1)
    if not readings:
        raise HTTPException(status_code=404, detail="No sensor readings recorded yet.")
    return with_predictions(readings)[0]


@app.get("/history")
def history(limit: int = Query(30, ge=1, le=1000)):
    """The last `limit` readings in chronological order (oldest first)."""
    readings = database.latest_readings(limit)
    readings.reverse()
    return with_predictions(readings)


@app.post("/feed", status_code=201)
def log_feed(feed: FeedEvent):
    """Log a batch of waste loaded now, with the plan for the next two batches."""
    return database.insert_feed(**feed.model_dump())


@app.get("/forecast")
def power_forecast():
    """Generator power expected 60 minutes after the latest reading (model trained on synthetic data)."""
    last = database.latest_timestamp()
    if last is None:
        return forecast(state["forecast_model"], [], [])
    since = (pd.Timestamp(last) - pd.Timedelta(minutes=RUN_AGE_CAP_MIN + 5)).strftime(database.TIME_FORMAT)
    return forecast(state["forecast_model"], database.readings_since(since), database.feeds_since(since))


@app.get("/forecast/metrics")
def forecast_metrics():
    if state["forecast_metrics"] is None:
        raise HTTPException(status_code=404, detail="Forecast metrics not found. Run: python -m wte.train_forecast")
    return state["forecast_metrics"]


@app.get("/integrity")
def integrity():
    """Re-verify the hash chain over every stored reading."""
    return database.verify_chain()


@app.get("/model/metrics")
def model_metrics():
    if state["metrics"] is None:
        raise HTTPException(status_code=404, detail="Model metrics not found. Run: python -m wte.train_model")
    return state["metrics"]
