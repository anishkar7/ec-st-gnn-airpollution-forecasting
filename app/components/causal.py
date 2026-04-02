import streamlit as st


def render_causal(dot_graph: str) -> None:
    st.markdown(
        "<span class='badge badge-placeholder'>Placeholder graph</span>",
        unsafe_allow_html=True,
    )
    st.subheader("Causal Dependency View")
    st.graphviz_chart(dot_graph)
    st.caption("Replace with NOTEARS/PCMCI output when model pipeline is connected.")
