import streamlit as st


def render_counterfactual(latest_pm25: float, counterfactual_fn) -> None:
    st.markdown(
        "<span class='badge badge-placeholder'>Scenario simulator</span>",
        unsafe_allow_html=True,
    )
    st.subheader("Counterfactual Simulation")
    traffic_cut = st.slider("Traffic Reduction (%)", min_value=0, max_value=50, value=20, step=5)
    industry_cut = st.slider("Industrial Emission Reduction (%)", min_value=0, max_value=50, value=10, step=5)

    projected_pm25, projected_drop = counterfactual_fn(latest_pm25, traffic_cut, industry_cut)
    st.metric("Projected PM2.5", f"{projected_pm25:.1f} ug/m3", f"-{projected_drop:.1f}% vs current")
