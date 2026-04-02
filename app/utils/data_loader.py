from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = REPO_ROOT / "data" / "processed"
METADATA_FILE = PROCESSED_DATA_DIR / "node_metadata.csv"
TIMESERIES_FILE = PROCESSED_DATA_DIR / "processed_delhi_data.csv"

REQUIRED_METADATA_COLUMNS = {"station", "latitude", "longitude", "station_id"}
REQUIRED_TIMESERIES_COLUMNS = {"datetime", "station", "pm25", "aqi"}


@st.cache_data(show_spinner=False)
def load_metadata() -> pd.DataFrame:
    metadata_df = pd.read_csv(METADATA_FILE)
    missing_columns = REQUIRED_METADATA_COLUMNS.difference(metadata_df.columns)
    if missing_columns:
        raise ValueError(f"node_metadata.csv missing columns: {sorted(missing_columns)}")
    return metadata_df.sort_values("station").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_timeseries() -> pd.DataFrame:
    timeseries_df = pd.read_csv(TIMESERIES_FILE, parse_dates=["datetime"])
    missing_columns = REQUIRED_TIMESERIES_COLUMNS.difference(timeseries_df.columns)
    if missing_columns:
        raise ValueError(f"processed_delhi_data.csv missing columns: {sorted(missing_columns)}")
    return timeseries_df.sort_values("datetime").reset_index(drop=True)


def station_history(timeseries_df: pd.DataFrame, station_name: str) -> pd.DataFrame:
    station_df = timeseries_df[timeseries_df["station"] == station_name].copy()
    if station_df.empty:
        raise ValueError(f"No timeseries rows found for station: {station_name}")
    return station_df.sort_values("datetime")


def station_snapshot(station_df: pd.DataFrame) -> tuple[float, float, float]:
    latest = station_df.iloc[-1]
    previous = station_df.iloc[-2] if len(station_df) > 1 else latest
    return float(latest["pm25"]), float(previous["pm25"]), float(latest["aqi"])


def build_forecast_frame(station_df: pd.DataFrame, forecast_horizon: int) -> pd.DataFrame:
    latest_window = station_df.tail(8)
    trend = latest_window["pm25"].diff().dropna().mean()
    trend = 0.0 if pd.isna(trend) else float(trend)

    last_timestamp = station_df.iloc[-1]["datetime"]
    last_value = float(station_df.iloc[-1]["pm25"])

    horizons = list(range(1, forecast_horizon + 1))
    forecast_values = [max(0.0, last_value + trend * h) for h in horizons]
    uncertainty = [max(8.0, abs(value) * 0.08) for value in forecast_values]

    return pd.DataFrame(
        {
            "datetime": [last_timestamp + pd.Timedelta(hours=h) for h in horizons],
            "pm25_forecast": forecast_values,
            "lower": [value - unc for value, unc in zip(forecast_values, uncertainty)],
            "upper": [value + unc for value, unc in zip(forecast_values, uncertainty)],
        }
    )
