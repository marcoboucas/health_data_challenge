import streamlit as st


def sidebar_view() -> None:
    """Document view."""

    st.sidebar.subheader("Settings")

    st.sidebar.slider("Number of patients", min_value=2, max_value=60, value=3, key="nbr-patients")

    # with st.sidebar.expander("Advanced Parameters"):
    #     st.slider("Number of clusters", min_value=1, max_value=3, key="nbr_clusters")
