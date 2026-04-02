import pandas as pd
import streamlit as st


def render_forecast(selected_station: str, forecast_df: pd.DataFrame, forecast_horizon: int) -> None:
    st.markdown("<span class='badge badge-real'>Real data</span>", unsafe_allow_html=True)
    st.subheader(f"Forecast for {selected_station}")

    chart_df = forecast_df.set_index("datetime")[["pm25_forecast", "lower", "upper"]]
    st.line_chart(chart_df, use_container_width=True)
    st.caption("pm25_forecast is trend-based baseline with uncertainty band for presentation readiness.")

    horizon_metric = forecast_df.iloc[-1]
    st.metric(
        f"Predicted PM2.5 (+{forecast_horizon}h)",
        f"{horizon_metric['pm25_forecast']:.1f} ug/m3",
        f"Range {horizon_metric['lower']:.1f} to {horizon_metric['upper']:.1f}",
    )
