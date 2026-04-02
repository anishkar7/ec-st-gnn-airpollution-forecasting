import pandas as pd
import streamlit as st


def render_attribution(attr_df: pd.DataFrame) -> None:
    st.markdown(
        "<span class='badge badge-placeholder'>Placeholder logic</span>",
        unsafe_allow_html=True,
    )
    st.subheader("Estimated Source Attribution")
    st.bar_chart(attr_df.set_index("Source"), use_container_width=True)
    st.dataframe(attr_df, use_container_width=True, hide_index=True)
