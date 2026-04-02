import pandas as pd
import streamlit as st

from utils.formatting import aqi_category


def render_overview(
    nodes_df: pd.DataFrame,
    timeseries_df: pd.DataFrame,
    latest_pm25: float,
    pm25_change: float,
    latest_aqi: float,
    pm25_pct_change: float,
) -> None:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            "Current PM2.5",
            f"{latest_pm25:.1f} ug/m3",
            f"{pm25_change:+.1f} ug/m3",
            delta_color="inverse",
        )
    with c2:
        st.metric("AQI Category", aqi_category(latest_aqi), f"AQI {latest_aqi:.0f}")
    with c3:
        st.metric("Short-Term Trend", f"{pm25_pct_change:+.1f}%", "vs previous reading")

    map_col, table_col = st.columns([2.2, 1.2])
    with map_col:
        st.subheader("Spatial Network")
        st.map(nodes_df[["latitude", "longitude"]], zoom=9)
    with table_col:
        st.subheader("Station Snapshot")
        latest_by_station = (
            timeseries_df.sort_values("datetime")
            .groupby("station", as_index=False)
            .tail(1)[["station", "pm25", "aqi"]]
            .sort_values("pm25", ascending=False)
            .head(8)
        )
        latest_by_station["AQI Band"] = latest_by_station["aqi"].apply(aqi_category)
        st.dataframe(latest_by_station, use_container_width=True, hide_index=True)
