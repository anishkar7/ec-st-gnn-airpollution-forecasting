import pandas as pd
import numpy as np
import streamlit as st

from components.attribution import render_attribution
from components.causal import render_causal
from components.counterfactual import render_counterfactual
from components.forecast import render_forecast
from components.overview import render_overview
from components.policy import render_policy
from components.styles import apply_theme

# Import the backend inference helpers.
from models.inference import (
    get_pm25_forecast,
    causal_graph_dot,
    policy_recommendation,
    run_counterfactual,
    source_attribution,
)
from utils.data_loader import (
    load_metadata,
    load_timeseries,
    station_history,
    station_snapshot,
)
from utils.formatting import safe_pct_change

# --- Helper: live AQI calculator ---
def calculate_indian_aqi(pm25):
    """Calculates dynamic Indian CPCB AQI based on raw PM2.5 to un-stick the UI."""
    if pm25 <= 30: return pm25 * (50/30)
    elif pm25 <= 60: return 50 + (pm25 - 30) * (50/30)
    elif pm25 <= 90: return 100 + (pm25 - 60) * (100/30)
    elif pm25 <= 120: return 200 + (pm25 - 90) * (100/30)
    elif pm25 <= 250: return 300 + (pm25 - 120) * (100/130)
    else: return min(500, 400 + (pm25 - 250) * (100/130)) # Keep the AQI within the top band.


st.set_page_config(page_title="Explainable ST-GNN Air Pollution Twin", layout="wide")
apply_theme()

try:
    # Load the dataset once at startup.
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

# Resolve the selected station to its exact spatial index.
# This keeps the dropdown aligned with the model columns.
station_idx = nodes_df.index[nodes_df['station'] == selected_station].tolist()[0]

station_df = station_history(timeseries_df, selected_station)

# Pull the latest readings and compute AQI dynamically.
latest_pm25, previous_pm25, _ = station_snapshot(station_df)
latest_aqi = calculate_indian_aqi(latest_pm25)

pm25_change = latest_pm25 - previous_pm25
pm25_pct_change = safe_pct_change(latest_pm25, previous_pm25)

# --- Direct model inference ---

# Work out how many 6-hour steps the slider requests.
num_steps = int(forecast_horizon // 6)

# 1. Run the ST-GNN on the preloaded dataframe.
forecast_matrix = get_pm25_forecast(df=timeseries_df) 

# 2. Slice the matrix for the selected station and horizon.
station_specific_forecast = forecast_matrix[:num_steps, station_idx] 

# 3. Build the chart dataframe dynamically.
last_time = pd.to_datetime(station_df['datetime'].iloc[-1])
future_times = [last_time + pd.Timedelta(hours=6 * (i + 1)) for i in range(num_steps)]

# Add simple confidence bounds for the UI.
lower_bounds = np.maximum(0, station_specific_forecast * 0.85)
upper_bounds = station_specific_forecast * 1.15

forecast_df = pd.DataFrame({
    'datetime': future_times,
    'pm25_forecast': station_specific_forecast, 
    'lower': lower_bounds,                      
    'upper': upper_bounds                       
})

# --- UI rendering ---
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
    # Pass the dataframe directly to avoid extra I/O.
    render_causal(causal_graph_dot(df=timeseries_df))

with tabs[4]:
    # Use a lambda to pass the dataframe into the component.
    render_counterfactual(latest_pm25, lambda pm, t, i: run_counterfactual(pm, t, i, df=timeseries_df))

with tabs[5]:
    # Render the policy recommendations as formatted markdown.
    render_policy(latest_pm25, policy_recommendation)

with st.expander("Methodology Snapshot"):
    st.write(
        "This dashboard operationalizes the final-year project narrative: spatio-temporal forecasting, "
        "source attribution, causal interpretability, counterfactual simulation, and policy support."
    )