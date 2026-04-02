import streamlit as st


def apply_theme() -> None:
    st.markdown(
        """
<style>
:root {
    --bg: #f3efe6;
    --panel: #ffffff;
    --ink: #111111;
    --muted: #5f6368;
    --accent: #e1552f;
    --accent-2: #1e6f86;
    --success: #2c7a3f;
    --warning: #b77700;
    --danger: #b42318;
}

.stApp, .stAppHeader {
    background:
        radial-gradient(circle at 8% 12%, #f9c59b 0%, rgba(249, 197, 155, 0) 45%),
        radial-gradient(circle at 85% 8%, #9bd3ff 0%, rgba(155, 211, 255, 0) 42%),
        var(--bg) !important;
}

h1, h2, h3, h4, h5, h6, p, label, span, li {
    color: var(--ink) !important;
}

h1 {
    letter-spacing: -0.8px !important;
    text-transform: uppercase !important;
    font-weight: 900 !important;
    border-bottom: 4px solid var(--ink) !important;
    padding-bottom: 8px !important;
    margin-bottom: 1rem !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ece7de 0%, #e6e0d5 100%) !important;
    border-right: 3px solid var(--ink) !important;
}

[data-testid="stMetric"] {
    background-color: var(--panel) !important;
    border: 2px solid var(--ink) !important;
    border-radius: 10px !important;
    box-shadow: 4px 4px 0 var(--ink) !important;
    padding: 14px !important;
    transition: transform 120ms ease, box-shadow 120ms ease;
}

[data-testid="stMetric"]:hover {
    transform: translate(-2px, -2px) !important;
    box-shadow: 6px 6px 0 var(--ink) !important;
}

[data-testid="stMetricLabel"] {
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
    color: #4e5359 !important;
    font-weight: 700 !important;
}

[data-testid="stMetricValue"] {
    font-weight: 900 !important;
    color: #111111 !important;
}

[data-testid="stMetricValue"] * {
    color: #111111 !important;
    opacity: 1 !important;
}

[data-testid="stMetricDelta"],
[data-testid="stMetricDelta"] * {
    opacity: 1 !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0.4rem;
}

.stTabs [data-baseweb="tab"] {
    background: #f7f3ea;
    border: 2px solid #111;
    border-radius: 8px 8px 0 0;
    padding: 0.45rem 0.75rem;
}

.stTabs [aria-selected="true"] {
    background: #fff0ea !important;
    color: var(--ink) !important;
    transform: translateY(-1px);
}

.block-note {
    background: rgba(255, 255, 255, 0.78);
    border-left: 5px solid var(--accent-2);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0 1rem 0;
    color: #1d242b !important;
}

.block-note * {
    color: #1d242b !important;
}

.badge {
    display: inline-block;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding: 0.22rem 0.52rem;
    border-radius: 999px;
    border: 1px solid #111;
    margin-right: 0.35rem;
}

.badge-real {
    background: #c8efd2;
}

.badge-placeholder {
    background: #ffe2cd;
}

[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    color: #40464d !important;
}

[data-testid="stDataFrame"] *,
[data-testid="stTable"] * {
    color: #111111 !important;
}
</style>
""",
        unsafe_allow_html=True,
    )
