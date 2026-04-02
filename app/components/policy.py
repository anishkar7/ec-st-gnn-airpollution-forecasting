import streamlit as st


def render_policy(latest_pm25: float, policy_fn) -> None:
    st.markdown(
        "<span class='badge badge-placeholder'>Policy heuristic</span>",
        unsafe_allow_html=True,
    )
    st.subheader("Policy Recommendation")

    recommendation, confidence = policy_fn(latest_pm25)
    st.success(recommendation)
    st.write(f"Estimated confidence: {confidence}")
