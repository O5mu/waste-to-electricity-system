"""Streamlit operations dashboard.

Run with:  streamlit run wte/dashboard.py
"""

import sys
from datetime import datetime
from pathlib import Path

# Allow `streamlit run wte/dashboard.py` from the project root (Streamlit only adds wte/ to the path).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

from wte.config import API_URL, PRESSURE_LIMIT_PSI, ROOT_DIR

STALE_AFTER_SECONDS = 30

COLORS = {
    "temperature": "#f97316",
    "pressure": "#eab308",
    "voltage": "#38bdf8",
    "power": "#10b981",
    "danger": "#ef4444",
    "text": "#cbd5e1",
    "muted": "#94a3b8",
    "grid": "rgba(148, 163, 184, 0.12)",
}

st.set_page_config(page_title="WTE Operations Dashboard", page_icon="⚡", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 2.5rem; max-width: 1400px; }
    .wte-eyebrow { color: #10b981; font-size: .78rem; font-weight: 600;
                   letter-spacing: .14em; text-transform: uppercase; margin-bottom: .2rem; }
    .wte-title { font-size: 2.1rem; font-weight: 700; line-height: 1.15; margin: 0; }
    .wte-sub { color: #94a3b8; margin: .35rem 0 1.25rem; }
    @media (max-width: 640px) { .wte-title { font-size: 1.6rem; } }

    .wte-status { display: flex; flex-wrap: wrap; align-items: center; gap: .75rem 1.5rem;
                  color: #94a3b8; font-size: .9rem; margin: .25rem 0 .75rem; }
    .pill { display: inline-flex; align-items: center; gap: .5rem; padding: .3rem .85rem;
            border-radius: 999px; font-size: .8rem; font-weight: 700; letter-spacing: .06em;
            border: 1px solid; }
    .pill .dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; }
    .pill.live { color: #34d399; background: rgba(16,185,129,.10); border-color: rgba(16,185,129,.35); }
    .pill.live .dot { animation: pulse 1.6s infinite; }
    .pill.stale { color: #fbbf24; background: rgba(234,179,8,.10); border-color: rgba(234,179,8,.35); }
    .pill.offline { color: #f87171; background: rgba(239,68,68,.10); border-color: rgba(239,68,68,.35); }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(52,211,153,.6); }
        70% { box-shadow: 0 0 0 8px rgba(52,211,153,0); }
        100% { box-shadow: 0 0 0 0 rgba(52,211,153,0); }
    }

    .banner { display: flex; align-items: center; gap: 1rem; padding: 1rem 1.25rem;
              border-radius: 12px; border: 1px solid; margin: .25rem 0 1.25rem; }
    .banner .icon { font-size: 1.6rem; line-height: 1; }
    .banner b { display: block; font-size: 1.02rem; margin-bottom: .1rem; }
    .banner span { color: #cbd5e1; font-size: .9rem; }
    .banner.ok { background: rgba(16,185,129,.07); border-color: rgba(16,185,129,.30); }
    .banner.ok b { color: #34d399; }
    .banner.danger { background: rgba(239,68,68,.14); border-color: rgba(239,68,68,.6);
                     animation: alarm 1s ease-in-out infinite alternate; }
    .banner.danger b { color: #fca5a5; }
    @keyframes alarm { from { box-shadow: 0 0 0 0 rgba(239,68,68,0); }
                       to { box-shadow: 0 0 22px 0 rgba(239,68,68,.35); } }

    [data-testid="stMetric"] { background: #131c2e; padding: 1rem 1.1rem; }
    [data-testid="stMetricLabel"] p { color: #94a3b8; font-size: .78rem; font-weight: 600;
                                      letter-spacing: .06em; text-transform: uppercase; }
    [data-testid="stMetricValue"] { font-weight: 700; }

    .section { font-size: 1.05rem; font-weight: 600; margin: 1.25rem 0 .1rem; }
    .caption { color: #94a3b8; font-size: .85rem; margin-bottom: .5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- API client ------------------------------------------------------------------------------

def api_get(path: str, **params):
    """GET a JSON resource from the backend; returns None if it is unreachable or errors."""
    try:
        response = requests.get(f"{API_URL}{path}", params=params, timeout=2)
        if response.ok:
            return response.json()
    except requests.RequestException:
        pass
    return None


# --- Chart helpers ---------------------------------------------------------------------------

def styled(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=36, b=28),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"], size=12),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(gridcolor=COLORS["grid"], zeroline=False, automargin=True)
    fig.update_yaxes(gridcolor=COLORS["grid"], zeroline=False, automargin=True)
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})


def break_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Insert empty rows where readings stop for a while, so lines don't bridge outages."""
    gap = df["timestamp"].diff()
    threshold = max(pd.Timedelta(seconds=15), gap.median() * 5)
    breaks = df.loc[gap > threshold, ["timestamp"]].copy()
    breaks["timestamp"] -= pd.Timedelta(milliseconds=1)
    return pd.concat([df, breaks]).sort_values("timestamp", kind="stable")


def pressure_gauge(pressure: float, limit: float) -> go.Figure:
    top = limit * 1.4
    critical = pressure >= limit
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pressure,
        number=dict(suffix=" PSI", font=dict(size=34, color=COLORS["danger"] if critical else "#f8fafc")),
        gauge=dict(
            axis=dict(range=[0, top], tickcolor=COLORS["muted"], tickfont=dict(color=COLORS["muted"])),
            bar=dict(color=COLORS["danger"] if critical else COLORS["pressure"], thickness=0.28),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            steps=[
                dict(range=[0, limit * 0.85], color="rgba(16,185,129,.16)"),
                dict(range=[limit * 0.85, limit], color="rgba(234,179,8,.20)"),
                dict(range=[limit, top], color="rgba(239,68,68,.24)"),
            ],
            threshold=dict(line=dict(color=COLORS["danger"], width=3), thickness=0.85, value=limit),
        ),
    ))
    fig.update_layout(height=290, margin=dict(l=24, r=24, t=24, b=8),
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(color=COLORS["text"]))
    return fig


def pressure_trend(df: pd.DataFrame, limit: float) -> go.Figure:
    critical = df[df["is_critical"]]
    df = break_gaps(df)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["pressure"], name="Pressure (PSI)", mode="lines",
        line=dict(color=COLORS["pressure"], width=2.5),
        fill="tozeroy", fillcolor="rgba(234,179,8,.08)",
    ))
    fig.add_trace(go.Scatter(
        x=critical["timestamp"], y=critical["pressure"], name="Over limit", mode="markers",
        marker=dict(color=COLORS["danger"], size=9, line=dict(color="#fff", width=1)),
    ))
    fig.add_hline(y=limit, line=dict(color=COLORS["danger"], dash="dash", width=1.5),
                  annotation_text=f"Safety limit · {limit:g} PSI",
                  annotation_font_color=COLORS["danger"], annotation_position="top left")
    fig.update_yaxes(ticksuffix=" PSI", range=[max(0, df["pressure"].min() - 5), max(df["pressure"].max(), limit) + 4])
    return styled(fig, height=290)


def temperature_voltage_trend(df: pd.DataFrame) -> go.Figure:
    df = break_gaps(df)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["temperature"], name="Temperature (°C)", mode="lines",
        line=dict(color=COLORS["temperature"], width=2.5),
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["voltage"], name="Voltage (V)", mode="lines",
        line=dict(color=COLORS["voltage"], width=2.5),
    ), secondary_y=True)
    fig.update_yaxes(ticksuffix=" °C", tickfont_color=COLORS["temperature"], secondary_y=False)
    fig.update_yaxes(ticksuffix=" V", tickformat=".2f", tickfont_color=COLORS["voltage"], showgrid=False, secondary_y=True)
    return styled(fig)


def power_trend(df: pd.DataFrame) -> go.Figure:
    df = break_gaps(df)
    fig = go.Figure(go.Scatter(
        x=df["timestamp"], y=df["predicted_power_w"], name="Estimated power (W)", mode="lines",
        line=dict(color=COLORS["power"], width=2.5),
        fill="tozeroy", fillcolor="rgba(16,185,129,.10)",
    ))
    fig.update_yaxes(ticksuffix=" W", range=[df["predicted_power_w"].min() * 0.97, df["predicted_power_w"].max() * 1.02])
    fig.update_layout(showlegend=False)
    return styled(fig)


# --- Sidebar ---------------------------------------------------------------------------------

health = api_get("/health")
pressure_limit = health["pressure_limit_psi"] if health else PRESSURE_LIMIT_PSI

with st.sidebar:
    st.markdown("### :material/bolt: WTE Control Panel")
    refresh_seconds = st.slider("Refresh interval (seconds)", 1, 10, 2)
    window = st.slider("Readings shown on charts", 20, 300, 60, step=10)

    st.divider()
    st.markdown("**Services**")
    if health:
        st.success("Backend API connected", icon="✅")
        if health["model_loaded"]:
            st.success("ML model loaded (Random Forest)", icon="🧠")
        else:
            st.warning("ML model missing. Run `python -m wte.train_model`.", icon="⚠️")
    else:
        st.error("Backend API unreachable", icon="🔌")
    st.caption(f"API: `{API_URL}` · [Open API docs]({API_URL}/docs)")
    st.caption(f"Safety limit: **{pressure_limit:g} PSI** chamber pressure")


# --- Header ----------------------------------------------------------------------------------

st.markdown(
    """
    <div class="wte-eyebrow">Senior Design Project · Edge Monitoring</div>
    <h1 class="wte-title">Integrated Waste-to-Electricity System</h1>
    <p class="wte-sub">Real-time monitoring, safety alerts and on-device ML power estimation
    for a steam-turbine waste-to-energy prototype.</p>
    """,
    unsafe_allow_html=True,
)

live_tab, model_tab, system_tab = st.tabs([":material/monitor_heart: Live Monitor", ":material/model_training: Model Performance", ":material/account_tree: System Architecture"])


# --- Live monitor ----------------------------------------------------------------------------

@st.fragment(run_every=refresh_seconds)
def live_monitor() -> None:
    history = api_get("/history", limit=window)

    if not history:
        reason = "The backend API is not reachable." if history is None else "No sensor readings have been recorded yet."
        st.markdown('<div class="wte-status"><span class="pill offline"><span class="dot"></span>OFFLINE</span></div>',
                    unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(f"#### Waiting for data\n{reason} Start the backend and a data source:")
            st.code("uvicorn wte.api:app --reload\n"
                    "python -m wte.simulator              # or: python -m wte.serial_bridge --port COM3",
                    language="bash")
        return

    df = pd.DataFrame(history)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    latest = df.iloc[-1]
    previous = df.iloc[-2] if len(df) > 1 else latest
    has_model = df["predicted_power_w"].notna().all()

    age = (datetime.now() - latest["timestamp"].to_pydatetime()).total_seconds()
    pill = ('<span class="pill live"><span class="dot"></span>LIVE</span>' if age <= STALE_AFTER_SECONDS
            else '<span class="pill stale"><span class="dot"></span>NO NEW DATA</span>')
    st.markdown(
        f'<div class="wte-status">{pill}'
        f'<span>Last reading <b>{latest["timestamp"]:%H:%M:%S}</b> · {latest["timestamp"]:%d %b %Y}</span>'
        f'<span>{len(df)} readings in view</span></div>',
        unsafe_allow_html=True,
    )

    # Safety banner
    if latest["is_critical"]:
        st.markdown(
            f'<div class="banner danger"><div class="icon">🚨</div><div>'
            f'<b>EMERGENCY: open the pressure relief valve</b>'
            f'<span>Chamber pressure is {latest["pressure"]:.1f} PSI, at or above the {pressure_limit:g} PSI safety limit.</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
    else:
        headroom = pressure_limit - latest["pressure"]
        st.markdown(
            f'<div class="banner ok"><div class="icon">🛡️</div><div>'
            f'<b>All systems nominal</b>'
            f'<span>Chamber pressure is {headroom:.1f} PSI below the {pressure_limit:g} PSI safety limit.</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Temperature", f"{latest['temperature']:.1f} °C",
              delta=f"{latest['temperature'] - previous['temperature']:+.1f} °C", border=True)
    c2.metric("Pressure", f"{latest['pressure']:.1f} PSI",
              delta=f"{latest['pressure'] - previous['pressure']:+.1f} PSI", delta_color="inverse", border=True)
    c3.metric("Voltage", f"{latest['voltage']:.2f} V",
              delta=f"{latest['voltage'] - previous['voltage']:+.2f} V", border=True)
    if has_model:
        c4.metric("Power · ML estimate", f"{latest['predicted_power_w']:.2f} W",
                  delta=f"{latest['predicted_power_w'] - previous['predicted_power_w']:+.2f} W", border=True)
    else:
        c4.metric("Power · ML estimate", "—", help="Model not loaded on the backend.", border=True)

    # Pressure safety
    st.markdown('<div class="section">Pressure safety</div>', unsafe_allow_html=True)
    gauge_col, trend_col = st.columns([1, 2], gap="medium")
    with gauge_col:
        with st.container(border=True):
            show(pressure_gauge(latest["pressure"], pressure_limit))
    with trend_col:
        with st.container(border=True):
            show(pressure_trend(df, pressure_limit))

    # Performance trends
    st.markdown('<div class="section">Performance trends</div>', unsafe_allow_html=True)
    left, right = st.columns(2, gap="medium")
    with left:
        with st.container(border=True):
            st.markdown('<div class="caption">Combustion temperature vs. generator voltage</div>', unsafe_allow_html=True)
            show(temperature_voltage_trend(df))
    with right:
        with st.container(border=True):
            st.markdown('<div class="caption">Generator power estimated by the ML model</div>',
                        unsafe_allow_html=True)
            if has_model:
                show(power_trend(df))
            else:
                st.info("Train the model to enable power estimation: `python -m wte.train_model`")


with live_tab:
    live_monitor()


# --- Model performance -----------------------------------------------------------------------

with model_tab:
    metrics = api_get("/model/metrics")
    if not metrics:
        st.warning("Model metrics are unavailable. Make sure the API is running and the model has been trained "
                   "with `python -m wte.train_model`.")
    else:
        st.markdown('<div class="section">Validation results on the held-out test set</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("R² score", f"{metrics['r2_score']:.3f}", border=True,
                  help="Share of the variance in power output explained by the model (1.0 = perfect).")
        m2.metric("Mean absolute error", f"{metrics['mae']:.3f} W", border=True)
        m3.metric("RMSE", f"{metrics['rmse']:.3f} W", border=True)
        m4.metric("Test samples", f"{metrics.get('test_samples', '—')}", border=True,
                  help="80 / 20 train-test split of the calibrated prototype dataset.")

        left, right = st.columns([3, 2], gap="medium")
        with left:
            with st.container(border=True):
                st.markdown("**Feature importance**")
                features = [f.split("_")[0] for f in metrics.get("features", ["Temperature_C", "Pressure_PSI"])]
                importance = metrics["feature_importance"]
                fig = go.Figure(go.Bar(
                    x=importance, y=features, orientation="h",
                    marker_color=[COLORS["temperature"], COLORS["pressure"]],
                    text=[f"{v:.1%}" for v in importance], textposition="outside", cliponaxis=False,
                ))
                fig.update_xaxes(tickformat=".0%", range=[0, 1])
                fig.update_layout(showlegend=False, hovermode=False)
                show(styled(fig, height=220))
                st.caption("Temperature dominates the prediction, consistent with the thermodynamics of the boiler.")
        with right:
            with st.container(border=True):
                st.markdown("**Model configuration**")
                params = {k.replace("rf__", ""): v for k, v in metrics["best_params"].items()}
                st.dataframe(
                    pd.DataFrame({"Setting": ["Algorithm", "Preprocessing", "Tuning", *params.keys(), "Trained on"],
                                  "Value": ["Random Forest Regressor", "StandardScaler", "GridSearchCV · 5-fold",
                                            *map(str, params.values()), str(metrics.get("training_date", "—")).replace("T", " ")]}),
                    hide_index=True, width="stretch",
                )
                st.caption("Inference runs locally on the backend; no cloud services are involved.")


# --- System architecture ---------------------------------------------------------------------

with system_tab:
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.image(str(ROOT_DIR / "docs" / "diagrams" / "architecture.png"), width="stretch",
                 caption="Edge computing architecture: all processing stays on the local host")
    with right:
        st.markdown(
            """
            #### How it works
            1. **Sensors** (K-type thermocouple, pressure transducer, voltage sensor) are read by an **Arduino**,
               which prints each sample as a JSON line over USB.
            2. The **serial bridge** forwards every line to the **FastAPI backend**, which validates it and stores it in **SQLite**.
            3. The backend runs the **Random Forest model** on each reading to estimate the generator's power output.
            4. This **dashboard** polls the API, raises a safety alert when chamber pressure reaches the limit,
               and charts the plant's behaviour in real time.

            Without hardware, the **simulator** generates realistic readings, including occasional pressure spikes.
            """
        )
    with st.expander("ML training pipeline"):
        st.image(str(ROOT_DIR / "docs" / "diagrams" / "ml-pipeline.png"), width="content")
