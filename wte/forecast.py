"""Features for the 60-minute-ahead power forecast, shared by training and the API.

Raw readings (every ~2 s) are averaged into 1-minute bins. Only the latest continuous stretch
of readings is used: a gap longer than `MAX_GAP_MIN` counts as a restart of the plant.
Each minute is joined with the operator's latest feed-log entry from the same run: what was
loaded, when, and the plan for the next batch.
"""

import numpy as np
import pandas as pd

from wte.plant import BURN_TAU_MIN, power_w

HORIZON_MIN = 60     # forecast this many minutes ahead
WINDOW_MIN = 30      # minutes of sensor history the features look back over
MAX_GAP_MIN = 2      # a longer break in readings starts a new run
RUN_AGE_CAP_MIN = 120

LAGS = [5, 10, 15, 20, 30]
SIGNALS = ["temperature", "pressure", "voltage", "power"]
FEED_COLUMNS = ["timestamp", "mass_kg", "next_feed_at", "next_mass_kg", "then_feed_at", "then_mass_kg"]


def to_minutes(readings: pd.DataFrame) -> pd.DataFrame:
    """1-minute means of the latest continuous run, with `run_minutes` since it started.

    `readings` has a datetime `timestamp` column plus temperature, pressure and voltage.
    """
    df = readings.set_index("timestamp")[["temperature", "pressure", "voltage"]].sort_index()
    minutes = df.resample("1min").mean().interpolate(limit=MAX_GAP_MIN, limit_area="inside")
    missing = minutes["temperature"].isna()
    if missing.any():
        minutes = minutes.loc[minutes.index > missing[missing].index[-1]]
    minutes["power"] = power_w(minutes["temperature"], minutes["pressure"])
    minutes["run_minutes"] = range(len(minutes))
    return minutes


def feeds_frame(feeds) -> pd.DataFrame:
    """Feed-log entries (records or a DataFrame) with datetime columns."""
    df = pd.DataFrame(feeds, columns=FEED_COLUMNS)
    for column in ("timestamp", "next_feed_at", "then_feed_at"):
        df[column] = pd.to_datetime(df[column]).astype("datetime64[ns]")
    return df


def make_features(minutes: pd.DataFrame, feeds) -> pd.DataFrame:
    """One feature row per minute. Rows without `WINDOW_MIN` of history, or without a feed
    logged in the current run, are NaN."""
    feeds = feeds_frame(feeds)
    features = {"run_minutes": minutes["run_minutes"].clip(upper=RUN_AGE_CAP_MIN)}
    for signal in SIGNALS:
        series = minutes[signal]
        features[f"{signal}_now"] = series
        for lag in LAGS:
            features[f"{signal}_lag{lag}"] = series.shift(lag)
        features[f"{signal}_mean15"] = series.rolling(15).mean()
        features[f"{signal}_mean30"] = series.rolling(30).mean()
        features[f"{signal}_std15"] = series.rolling(15).std()
    for span in (5, 15):
        features[f"temperature_slope{span}"] = (minutes["temperature"] - minutes["temperature"].shift(span)) / span
    out = pd.DataFrame(features, index=minutes.index)

    # Latest feed logged by the end of each minute bin.
    bin_end = pd.DataFrame({"bin_end": (minutes.index + pd.Timedelta(minutes=1)).astype("datetime64[ns]")})
    log = pd.merge_asof(bin_end, feeds[FEED_COLUMNS].sort_values("timestamp"),
                        left_on="bin_end", right_on="timestamp", direction="backward")
    since_feed = (log["bin_end"] - log["timestamp"]).dt.total_seconds().to_numpy() / 60
    in_run = since_feed <= minutes["run_minutes"].to_numpy() + 1
    out["minutes_since_feed"] = since_feed
    out["last_mass_kg"] = log["mass_kg"].to_numpy()
    for batch in ("next", "then"):
        out[f"minutes_to_{batch}_feed"] = (log[f"{batch}_feed_at"] - log["bin_end"]).dt.total_seconds().to_numpy() / 60
        out[f"{batch}_mass_kg"] = log[f"{batch}_mass_kg"].to_numpy()

    # Unburnt waste in the chamber, assuming each batch burns off exponentially (BURN_TAU_MIN):
    # now and at points over the next hour, from the batches logged so far plus the planned ones.
    run_start = bin_end["bin_end"].min() - pd.Timedelta(minutes=2)
    run_feeds = feeds[feeds["timestamp"] >= run_start]
    ages = (bin_end["bin_end"].to_numpy()[:, None] - run_feeds["timestamp"].to_numpy()[None, :]) / np.timedelta64(1, "m")
    masses = run_feeds["mass_kg"].to_numpy()[None, :]
    for ahead in (0, 15, 30, 45, 60):
        waste = np.where(ages >= 0, masses * np.exp(-(ages + ahead) / BURN_TAU_MIN), 0.0).sum(axis=1)
        for batch in ("next", "then"):
            due = out[f"minutes_to_{batch}_feed"].to_numpy()
            mass = out[f"{batch}_mass_kg"].to_numpy()
            waste += np.where(due <= ahead, mass * np.exp(-(ahead - np.maximum(due, 0)) / BURN_TAU_MIN), 0.0)
        out[f"waste_kg_in{ahead}"] = waste

    out.loc[(minutes["run_minutes"] < WINDOW_MIN).to_numpy() | ~in_run] = float("nan")
    return out


FEATURE_NAMES = list(make_features(
    pd.DataFrame({"temperature": [], "pressure": [], "voltage": [], "power": [], "run_minutes": []},
                 index=pd.DatetimeIndex([])),
    [],
).columns)


def forecast(model, readings: list[dict], feeds: list[dict]) -> dict:
    """Forecast power `HORIZON_MIN` minutes after the latest reading."""
    result = {"available": False, "reason": None, "history_minutes": 0, "required_minutes": WINDOW_MIN,
              "horizon_minutes": HORIZON_MIN, "predicted_power_w": None, "target_time": None}
    if model is None:
        result["reason"] = "Forecast model not trained. Run: python -m wte.train_forecast"
        return result
    if not readings:
        result["reason"] = "No sensor readings recorded yet."
        return result

    df = pd.DataFrame(readings)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    minutes = to_minutes(df)
    result["history_minutes"] = int(minutes["run_minutes"].iloc[-1])
    result["target_time"] = (df["timestamp"].max() + pd.Timedelta(minutes=HORIZON_MIN)).strftime("%Y-%m-%d %H:%M:%S")

    row = make_features(minutes, feeds).iloc[[-1]][FEATURE_NAMES]
    if result["history_minutes"] < WINDOW_MIN:
        result["reason"] = f"Needs {WINDOW_MIN} min of continuous readings ({result['history_minutes']} so far)."
    elif row.isna().any(axis=None):
        result["reason"] = "No feed logged since the plant started. Log feeds with POST /feed."
    else:
        result["available"] = True
        result["predicted_power_w"] = round(float(model.predict(row)[0]), 4)
    return result
