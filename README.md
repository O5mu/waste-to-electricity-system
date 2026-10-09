# Integrated Waste-to-Electricity System

Real-time monitoring, safety alerting and on-device machine-learning power estimation for a
steam-turbine **waste-to-energy (WTE)** prototype, built as a senior design project.

Sensors on the boiler stream temperature, pressure and voltage through an Arduino to a local
FastAPI backend. A Random Forest model estimates the generator's power output from every reading,
and a Streamlit dashboard shows everything live, raising an alarm the moment chamber pressure
reaches the safety limit. Everything runs on the local machine, with no cloud services involved.

![Live monitor](docs/screenshots/live-monitor.png)

## Features

- **Live monitoring** of combustion temperature, chamber pressure and generator voltage
- **Safety alerting**: a pressure gauge and an emergency banner when pressure reaches the limit (60 PSI by default)
- **ML power estimation**: a Random Forest regressor (R² = 0.911, MAE = 0.091 W) predicts power output from temperature and pressure
- **Model audit view** with validation metrics, feature importance and tuned hyperparameters
- **Hardware or simulation**: run against the real Arduino rig, or use the built-in sensor simulator
- **Edge architecture**: SQLite storage and local inference with no external dependencies

## Architecture

![Architecture](docs/diagrams/architecture.png)

| Component | File | Role |
|---|---|---|
| Serial bridge | [`wte/serial_bridge.py`](wte/serial_bridge.py) | Reads JSON lines from the Arduino over USB and forwards them to the API |
| Simulator | [`wte/simulator.py`](wte/simulator.py) | Generates realistic readings (with optional pressure spikes) when no hardware is attached |
| Backend API | [`wte/api.py`](wte/api.py) | Validates and stores readings in SQLite; serves them with ML power estimates |
| Dashboard | [`wte/dashboard.py`](wte/dashboard.py) | Streamlit operations dashboard |
| Model training | [`wte/train_model.py`](wte/train_model.py) | Generates the calibrated dataset, tunes and trains the model |
| Configuration | [`wte/config.py`](wte/config.py) | Paths, API URL and safety limit (overridable with environment variables) |

More diagrams (class, sequence, use-case, ML pipeline) are in [`docs/diagrams`](docs/diagrams).

## Getting started

Requires **Python 3.10+**.

```bash
git clone https://github.com/O5mu/waste-to-electricity-system.git
cd waste-to-electricity-system
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Run each of these in its own terminal from the project root:

```bash
# 1. Backend API (docs at http://127.0.0.1:8000/docs)
uvicorn wte.api:app --reload

# 2. A data source: the simulator...
python -m wte.simulator
#    ...or the real hardware
python -m wte.serial_bridge --port COM3

# 3. Dashboard (opens http://localhost:8501)
streamlit run wte/dashboard.py
```

A trained model is included in [`models/`](models). To retrain it:

```bash
python -m wte.train_model
```

### Simulator options

| Option | Default | Description |
|---|---|---|
| `--interval` | `2` | Seconds between readings |
| `--fault-rate` | `0.1` | Probability of a high-pressure spike (`0` = stable operation) |
| `--api-url` | `http://127.0.0.1:8000` | Backend URL |

### Configuration

| Environment variable | Default | Description |
|---|---|---|
| `WTE_API_URL` | `http://127.0.0.1:8000` | Backend URL used by the dashboard, simulator and bridge |
| `WTE_PRESSURE_LIMIT_PSI` | `60` | Chamber pressure that triggers the safety alert |
| `WTE_DB_PATH` | `data/wte_system.db` | SQLite database location |

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service status, model status and safety limit |
| `POST` | `/update-reading` | Store a reading: `{"temperature": 101.2, "pressure": 55.4, "voltage": 13.6}` |
| `GET` | `/latest-reading` | Most recent reading with `predicted_power_w` and `is_critical` |
| `GET` | `/history?limit=30` | Last *N* readings (oldest first) with predictions |
| `GET` | `/model/metrics` | Validation metrics of the trained model |

## Hardware

| Sensor | Measures |
|---|---|
| K-type thermocouple | Combustion temperature (°C) |
| Pressure transducer | Chamber pressure (PSI) |
| Voltage sensor | Generator output voltage (V) |

The Arduino prints one JSON object per line at 115200 baud, for example
`{"temperature": 101.2, "pressure": 55.4, "voltage": 13.6}`.

## Machine-learning model

The model is a `StandardScaler` → `RandomForestRegressor` pipeline, tuned with 5-fold `GridSearchCV`
on 1,200 samples calibrated to the prototype's ~6 W generator (temperature 94–118 °C,
pressure 50–80 PSI) and evaluated on a 20 % hold-out set.

| Metric | Value |
|---|---|
| R² | 0.911 |
| MAE | 0.091 W |
| RMSE | 0.114 W |
| Feature importance | Temperature 75 % · Pressure 25 % |

![ML pipeline](docs/diagrams/ml-pipeline.png)

## Project structure

```
├── wte/                 # Application package
│   ├── api.py           # FastAPI backend
│   ├── dashboard.py     # Streamlit dashboard
│   ├── database.py      # SQLite access
│   ├── simulator.py     # Sensor simulator
│   ├── serial_bridge.py # Arduino → API bridge
│   ├── train_model.py   # Dataset generation & model training
│   └── config.py        # Settings
├── models/              # Trained model + validation metrics
├── data/                # Training dataset (SQLite DB is created here at runtime)
├── docs/                # Diagrams and screenshots
└── .streamlit/          # Dashboard theme
```
