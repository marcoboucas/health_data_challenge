"""Demonstrator."""
# pylint: disable=C0413

import sys

import pandas as pd
import streamlit as st

sys.path.append(".")
from demonstrator.expandable import expandable_cluster

# pylint: disable=wrong-import-position
from demonstrator.sidebar import filter_data, sidebar_filters, sidebar_view
from demonstrator.uploader import run_pipeline, uploader
from demonstrator.utils import load_fake_data

st.set_page_config(
    page_title="Health Challenge",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://www.extremelycoolapp.com/help",
        "Report a bug": "https://www.extremelycoolapp.com/bug",
        "About": "# This is a header. This is an *extremely* cool app!",
    },
)
st.markdown("## Health Project❤️")


@st.cache
def load_data(fake: bool = True):
    """Load the data."""
    if fake:
        return load_fake_data()
    return run_pipeline(st.session_state.get("data-folder", "./data/val/txt/"))


is_fake_data = st.session_state.get("is-fake-data", True)

st.markdown("### Upload data")

uploader()


@st.cache
def to_csv(df):
    """To CSV"""
    return df.to_csv().encode("utf-8")


sidebar_view()


data = load_data(fake=is_fake_data)

filters = sidebar_filters(data)

data = filter_data(data, filters)


# Add new filters based on the data we have

st.markdown("### Cohort Selection")

for cluster_id, cluster in data.items():
    st.checkbox(
        f"({len(cluster['patients'])}) {cluster['name']}",
        key=f"pin-{cluster.get('name', 'name')}",
        value=cluster_id == "1",
    )


cohortes_df_ = []
for cluster in data.values():
    if st.session_state.get(f"pin-{cluster['name']}") is True:
        cohortes_df_.extend(cluster["patients"])

st.metric(
    "Number of patients",
    len(cohortes_df_),
    len(cohortes_df_) - st.session_state.get("nbr-patients"),
)
st.download_button(
    "Download Excel 📂", to_csv(pd.DataFrame.from_records(cohortes_df_)), file_name="cohorts.csv"
)


for cluster in data.values():
    if st.session_state.get(f"pin-{cluster['name']}") is True:
        expandable_cluster(cluster)
