"""Demonstrator."""


import sys

import streamlit as st

sys.path.append(".")
# pylint: disable=wrong-import-position
from demonstrator.document_view import document_view
from demonstrator.utils import load_models, update_document
from src.dataset.dataset_loader import DatasetLoader

st.set_page_config(
    page_title="Health Challenge",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://www.extremelycoolapp.com/help",
        "Report a bug": "https://www.extremelycoolapp.com/bug",
        "About": "# This is a header. This is an *extremely* cool app!",
    },
)
st.markdown("## Health Project")

with st.expander("Model parameters", False):
    with st.form("params"):
        st.selectbox("NER Model", options=["RegexNer", "MedCATNer"], index=0, key="ner_model")
        st.form_submit_button(on_click=load_models)


if "dataset" not in st.session_state:
    st.session_state["dataset"] = DatasetLoader(mode="test")

if "ner" not in st.session_state:
    load_models()


with st.container():
    st.markdown(f"Model: {st.session_state.get('ner').__class__.__name__}")


st.select_slider(
    label="Select the document you want to examine",
    options=range(len(st.session_state["dataset"])),
    value=0,
    key="dataset_instance_id",
    on_change=update_document,
)

if "dataset_instance" in st.session_state:

    document_view()

else:
    update_document()
