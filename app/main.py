import streamlit as st
import pandas as pd
import numpy as np

# --- Page Config ---
st.set_page_config(page_title="Air Pollution Digital Twin", layout="wide")

# --- SURGICAL GUMROAD CSS ---
st.markdown("""
<style>
    /* 1. App Background */
    .stApp, .stAppHeader {
        background-color: #F4F0EA !important; 
    }
    
    /* 2. Global Typography (Only target headers and labels, NOT complex divs) */
    h1, h2, h3, h4, h5, h6, label, p {
        color: #111111 !important;
        font-family: 'Inter', 'Helvetica Neue', sans-serif !important;
    }

    /* 3. Main Title Styling */
    h1 {
        font-weight: 900 !important;
        text-transform: uppercase !important;
        letter-spacing: -1px !important;
        border-bottom: 4px solid #111111 !important;
        padding-bottom: 10px !important;
        margin-bottom: 30px !important;
    }

    /* 4. Metric Cards (The Gumroad Tactile Look) */
    [data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 3px solid #111111 !important;
        padding: 20px !important;
        border-radius: 0px !important; /* Sharp corners */
        box-shadow: 6px 6px 0px #111111 !important;
        transition: all 0.15s ease-in-out !important;
    }
    
    /* Card Hover Animation */
    [data-testid="stMetric"]:hover {
        transform: translate(-3px, -3px) !important;
        box-shadow: 9px 9px 0px #111111 !important;
    }

    /* 5. Metric Text Visibility */
    [data-testid="stMetricLabel"] {
        font-weight: 800 !important;
        text-transform: uppercase !important;
        font-size: 13px !important;
        letter-spacing: 1px !important;
        color: #555555 !important;
    }
    
    [data-testid="stMetricValue"] {
        font-weight: 900 !important;
        font-size: 48px !important;
        color: #111111 !important;
    }

    /* 6. Sidebar Background */
    [data-testid="stSidebar"] {
        background-color: #E8E4DF !important;
        border-right: 4px solid #111111 !important;
    }

    /* 7. Input Fields (Borders only, let Streamlit handle text) */
    div[data-baseweb="select"] > div, div[data-baseweb="base-input"] > input {
        border: 2px solid #111111 !important;
        border-radius: 0px !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Data ---
@st.cache_data
def load_metadata():
    return pd.read_csv(r"C:\Users\ANISH\Desktop\FRP\ecstgnn\ec-st-gnn-airpollution-forecasting\data\processed\node_metadata.csv")

nodes_df = load_metadata()

# --- Header ---
st.title("🏙️ ST-GNN Digital Twin")

# --- Sidebar Controls ---
st.sidebar.markdown("## ⚙️ CONTROLS")
selected_station = st.sidebar.selectbox("TARGET STATION", nodes_df['station'])
forecast_horizon = st.sidebar.slider("FORECAST HORIZON (HRS)", min_value=6, max_value=24, step=6)

# --- Top Row: Metrics ---
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.metric(label=f"CURRENT PM2.5: {selected_station}", value="185 µg", delta="12 µg (Rising)", delta_color="inverse")
with col2:
    st.metric(label=f"PREDICTED (+{forecast_horizon}H)", value="210 µg", delta="Hazardous Alert", delta_color="inverse")
with col3:
    st.metric(label="PRIMARY CAUSAL SOURCE", value="Upstream", delta="Punjab / Haryana", delta_color="normal")

st.markdown("<br><br>", unsafe_allow_html=True)

# --- Bottom Row: Map and Chart ---
col_map, col_chart = st.columns([2, 1.5])

with col_map:
    st.markdown("### 📍 SPATIAL NETWORK (DELHI)")
    st.map(nodes_df[['latitude', 'longitude']], zoom=10)

with col_chart:
    st.markdown("### 📈 TEMPORAL FORECAST")
    # Mock dataframe for the line chart
    mock_time = pd.date_range(start=pd.Timestamp.now(), periods=forecast_horizon, freq='H')
    mock_pm25 = np.random.randint(150, 250, size=forecast_horizon)
    chart_df = pd.DataFrame({"PM2.5 Level": mock_pm25}, index=mock_time)
    
    # Streamlit native chart - Using our accent color
    st.line_chart(chart_df, color="#FF4B4B")