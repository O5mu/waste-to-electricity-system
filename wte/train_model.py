"""Generates the calibrated prototype dataset and trains the power-output model.

Run with:  python -m wte.train_model
Outputs:   data/wte_simulation_data.csv, models/wte_model.joblib, models/model_metrics.json
"""

import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from wte.config import DATASET_PATH, FEATURES, METRICS_PATH, MODEL_PATH, TARGET

SEED = 42


def generate_dataset(samples: int = 1200) -> pd.DataFrame:
    """Synthetic boiler data calibrated to the ~6 W prototype generator."""
    np.random.seed(SEED)  # legacy seeding keeps the dataset identical to the one in the report
    temperature = np.random.uniform(94, 118, samples)  # °C
    pressure = np.random.uniform(50, 80, samples)      # PSI
    power = 0.045 * temperature + 0.02 * pressure + np.random.normal(0, 0.1, samples)  # W

    df = pd.DataFrame({FEATURES[0]: temperature, FEATURES[1]: pressure, TARGET: power})
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATASET_PATH, index=False)
    print(f"Dataset: {samples} samples -> {DATASET_PATH}")
    return df


def train() -> None:
    df = generate_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=SEED
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(random_state=SEED)),
    ])
    param_grid = {
        "rf__n_estimators": [100, 200],
        "rf__max_depth": [None, 10, 20],
        "rf__min_samples_split": [2, 5],
    }

    print("Tuning hyperparameters (5-fold grid search)...")
    search = GridSearchCV(pipeline, param_grid, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1)
    search.fit(X_train, y_train)
    model = search.best_estimator_

    y_pred = model.predict(X_test)
    results = {
        "r2_score": metrics.r2_score(y_test, y_pred),
        "mae": metrics.mean_absolute_error(y_test, y_pred),
        "rmse": float(np.sqrt(metrics.mean_squared_error(y_test, y_pred))),
        "features": FEATURES,
        "feature_importance": model.named_steps["rf"].feature_importances_.tolist(),
        "best_params": search.best_params_,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "training_date": datetime.now().isoformat(timespec="seconds"),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(results, indent=2))

    print(f"Best params: {search.best_params_}")
    print(f"R²: {results['r2_score']:.4f} | MAE: {results['mae']:.4f} W | RMSE: {results['rmse']:.4f} W")
    print(f"Saved model -> {MODEL_PATH}")


if __name__ == "__main__":
    train()
