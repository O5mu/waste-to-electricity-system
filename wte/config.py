"""Central configuration. Every value can be overridden with an environment variable."""

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

DB_PATH = Path(os.getenv("WTE_DB_PATH", DATA_DIR / "wte_system.db"))
DATASET_PATH = DATA_DIR / "wte_simulation_data.csv"
MODEL_PATH = MODELS_DIR / "wte_model.joblib"
METRICS_PATH = MODELS_DIR / "model_metrics.json"

API_URL = os.getenv("WTE_API_URL", "http://127.0.0.1:8000").rstrip("/")

# Chamber pressure at or above this value triggers the safety alert.
PRESSURE_LIMIT_PSI = float(os.getenv("WTE_PRESSURE_LIMIT_PSI", "60"))

# Model features, in the order the model was trained on.
FEATURES = ["Temperature_C", "Pressure_PSI"]
TARGET = "Power_Output_W"
