"""Trains the 60-minute-ahead power forecast on synthetic plant runs.

Run with:  python -m wte.train_forecast
Outputs:   models/forecast_model.joblib, models/forecast_metrics.json

The training data is simulated by `wte.plant` (no real plant data exists yet). Runs are split
in time order: the model is tuned and trained on the earlier runs and scored on the later ones.
"""

import json
import random
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from wte.config import FORECAST_METRICS_PATH, FORECAST_MODEL_PATH
from wte.forecast import FEATURE_NAMES, HORIZON_MIN, WINDOW_MIN, make_features, to_minutes
from wte.plant import PlantProcess

SEED = 42
RUNS = 40
INTERVAL_S = 2.0
TEST_SHARE = 0.2


def simulate_run(rng: random.Random, start: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Readings every `INTERVAL_S` seconds for one run (plus the plant's true power), and its feed log."""
    plant = PlantProcess(rng, fault_rate=rng.uniform(0.0, 0.15))
    steps = int(rng.uniform(6, 14) * 3600 / INTERVAL_S)
    rows, feeds = [], [dict(f, timestamp=start) for f in plant.pop_feeds()]
    for step in range(1, steps + 1):
        reading = plant.read(INTERVAL_S)
        reading["true_power"] = plant.true_power()
        reading["timestamp"] = start + pd.Timedelta(seconds=step * INTERVAL_S)
        rows.append(reading)
        feeds += [dict(f, timestamp=reading["timestamp"]) for f in plant.pop_feeds()]
    log = pd.DataFrame(feeds)
    for batch in ("next", "then"):
        log[f"{batch}_feed_at"] = log["timestamp"] + pd.to_timedelta(log[f"{batch}_feed_in_min"], unit="min")
    return pd.DataFrame(rows), log


def build_dataset() -> pd.DataFrame:
    """Feature rows with the true power `HORIZON_MIN` minutes later, for every run."""
    rng = random.Random(SEED)
    start = pd.Timestamp("2026-01-05 06:00")
    frames = []
    for run in range(RUNS):
        readings, feeds = simulate_run(rng, start)
        minutes = to_minutes(readings)
        frame = make_features(minutes, feeds)
        frame["minute"] = minutes["run_minutes"]
        frame["power_now"] = minutes["power"]
        frame["target"] = readings.set_index("timestamp")["true_power"].resample("1min").mean().shift(-HORIZON_MIN)
        frame["run"] = run
        frames.append(frame.dropna())
        start = readings["timestamp"].iloc[-1].ceil("D") + pd.Timedelta(hours=6)
        print(f"  run {run + 1:2d}/{RUNS}: {len(minutes) / 60:4.1f} h")
    return pd.concat(frames)


def train() -> None:
    print(f"Simulating {RUNS} plant runs...")
    data = build_dataset()
    test_runs = sorted(data["run"].unique())[-int(RUNS * TEST_SHARE):]
    train_set, test_set = data[~data["run"].isin(test_runs)], data[data["run"].isin(test_runs)]

    param_grid = {
        "n_estimators": [200],
        "max_depth": [12, 20],
        "min_samples_leaf": [5, 20],
        "max_features": [0.33, 1.0],
    }
    print("Tuning hyperparameters (time-ordered 4-fold CV)...")
    search = GridSearchCV(
        RandomForestRegressor(random_state=SEED),
        param_grid,
        cv=TimeSeriesSplit(n_splits=4, gap=HORIZON_MIN),
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    search.fit(train_set[FEATURE_NAMES], train_set["target"])
    model = search.best_estimator_

    y_test = test_set["target"]
    y_pred = model.predict(test_set[FEATURE_NAMES])
    persistence = test_set["power_now"]  # "power in an hour = power now", the no-forecast baseline

    importance = sorted(zip(FEATURE_NAMES, model.feature_importances_), key=lambda x: -x[1])
    sample = test_set[test_set["run"] == test_runs[0]]
    results = {
        "r2_score": metrics.r2_score(y_test, y_pred),
        "mae": metrics.mean_absolute_error(y_test, y_pred),
        "rmse": float(np.sqrt(metrics.mean_squared_error(y_test, y_pred))),
        "persistence_r2": metrics.r2_score(y_test, persistence),
        "persistence_mae": metrics.mean_absolute_error(y_test, persistence),
        "horizon_minutes": HORIZON_MIN,
        "window_minutes": WINDOW_MIN,
        "data": "synthetic (wte.plant simulation)",
        "top_features": [{"feature": f, "importance": float(v)} for f, v in importance[:10]],
        "best_params": search.best_params_,
        "train_runs": RUNS - len(test_runs),
        "test_runs": len(test_runs),
        "train_samples": len(train_set),
        "test_samples": len(test_set),
        "sample_run": {
            "target_minute": (sample["minute"] + HORIZON_MIN).tolist(),  # minutes since the run started
            "actual": sample["target"].round(4).tolist(),
            "forecast": np.round(model.predict(sample[FEATURE_NAMES]), 4).tolist(),
            "persistence": sample["power_now"].round(4).tolist(),
        },
        "training_date": datetime.now().isoformat(timespec="seconds"),
    }

    FORECAST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, FORECAST_MODEL_PATH, compress=3)
    FORECAST_METRICS_PATH.write_text(json.dumps(results, indent=2))

    print(f"Best params: {search.best_params_}")
    print(f"Forecast    R²: {results['r2_score']:.4f} | MAE: {results['mae']:.4f} W | RMSE: {results['rmse']:.4f} W")
    print(f"Persistence R²: {results['persistence_r2']:.4f} | MAE: {results['persistence_mae']:.4f} W")
    print(f"Saved model -> {FORECAST_MODEL_PATH}")


if __name__ == "__main__":
    train()
