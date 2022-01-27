from collections import Counter
from copy import deepcopy
from typing import Dict

import streamlit as st

categories = ["treatments", "problems", "tests"]


def sidebar_view() -> None:
    """Document view."""

    st.sidebar.subheader("Settings")

    st.sidebar.slider("Number of patients", min_value=2, max_value=60, value=42, key="nbr-patients")

    # with st.sidebar.expander("Advanced Parameters"):
    #     st.slider("Number of clusters", min_value=1, max_value=3, key="nbr_clusters")


def sidebar_filters(data):
    """Sidebar filters, return the values of the different filters."""

    filters: Dict[str, str] = {}
    with st.sidebar.expander("Filters: Problems"):
        counter = Counter()
        for cluster in data.values():
            for patient in cluster["patients"]:
                counter.update(list(patient["problems"].keys()))

        for type_name, count in counter.most_common(5):
            filters[type_name] = st.selectbox(
                f"{type_name} ({count})", ["all", "present", "absent", "hypothetical"], index=0
            )
    return filters


@st.cache
def filter_data(data, filters):
    """Filter the data."""
    new_data = deepcopy(data)

    for cluster in new_data.values():
        new_patients = []
        for patient in cluster["patients"]:
            for filter_name, filter_value in filters.items():
                if filter_name in patient["problems"]:
                    if filter_value not in ["all", patient["problems"][filter_name]]:
                        break
            else:
                new_patients.append(patient)
        cluster["patients"] = new_patients
    return new_data
