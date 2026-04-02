import pandas as pd
import streamlit as st

from components.attribution import render_attribution
from components.causal import render_causal
from components.counterfactual import render_counterfactual
from components.forecast import render_forecast
from components.overview import render_overview
from components.policy import render_policy
from components.styles import apply_theme
from models.inference import (
    causal_graph_dot,
    policy_recommendation,
    run_counterfactual,
    source_attribution,
)
from utils.data_loader import (
    build_forecast_frame,
    load_metadata,
    load_timeseries,
    station_history,
    station_snapshot,
)
from utils.formatting import safe_pct_change

st.set_page_config(page_title="Explainable ST-GNN Air Pollution Twin", layout="wide")
apply_theme()

try:
    nodes_df = load_metadata()
    timeseries_df = load_timeseries()
except Exception as exc:
    st.error(f"Data initialization failed: {exc}")
    st.info("Verify files exist inside data/processed and schema has required columns.")
    st.stop()

st.title("ST-GNN Digital Twin for Delhi-NCR")
st.markdown(
    "<div class='block-note'>"
    "Live dashboard for final-year project demonstration: forecasting, explainability, causal reasoning, and policy simulation."
    "</div>",
    unsafe_allow_html=True,
)

st.sidebar.markdown("## Control Panel")
selected_station = st.sidebar.selectbox("Target Station", nodes_df["station"].tolist(), index=0)
forecast_horizon = st.sidebar.slider("Forecast Horizon (hours)", min_value=6, max_value=24, value=24, step=6)
st.sidebar.caption("Data source: processed_delhi_data.csv + node_metadata.csv")

station_df = station_history(timeseries_df, selected_station)
latest_pm25, previous_pm25, latest_aqi = station_snapshot(station_df)
pm25_change = latest_pm25 - previous_pm25
pm25_pct_change = safe_pct_change(latest_pm25, previous_pm25)

# --- CLEAN ARCHITECTURE ---
# The AI Model is now safely hidden inside this function call!
forecast_df = build_forecast_frame(station_df, forecast_horizon)

tabs = st.tabs(
    [
        "Overview",
        "Forecast",
        "Source Attribution",
        "Causal Graph",
        "Counterfactual Lab",
        "Policy Recommendation",
    ]
)

with tabs[0]:
    render_overview(nodes_df, timeseries_df, latest_pm25, pm25_change, latest_aqi, pm25_pct_change)

with tabs[1]:
    render_forecast(selected_station, forecast_df, forecast_horizon)

with tabs[2]:
    attr_df = source_attribution(station_df.iloc[-1])
    render_attribution(attr_df)

with tabs[3]:
    render_causal(causal_graph_dot())

with tabs[4]:
    render_counterfactual(latest_pm25, run_counterfactual)

with tabs[5]:
    render_policy(latest_pm25, policy_recommendation)

with st.expander("Methodology Snapshot"):
    st.write(
        "This dashboard operationalizes the final-year project narrative: spatio-temporal forecasting, "
        "source attribution, causal interpretability, counterfactual simulation, and policy support."
    )