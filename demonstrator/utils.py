"""Utils."""

import html
from typing import List

import streamlit as st
from annotated_text import HtmlElement, annotation, div

from src import config
from src.dataset.dataset_loader import DataInstance
from src.models.medcat_ner import MedCATNer
from src.models.regex_ner import RegexNer


def display_annotations(ents: List):
    """Display the annotations.

    From annotated_text.py:annotated_text() but without
    the problem with several lines."""
    out = div()
    for arg in ents:
        if isinstance(arg, str):
            out(html.escape(arg))
        elif isinstance(arg, HtmlElement):
            out(arg)
        elif isinstance(arg, tuple):
            out(annotation(*arg))
    st.markdown(str(out).strip("\n").replace("\n", "<br/>"), unsafe_allow_html=True)


def update_document():
    """Update the selected document."""
    with st.spinner("Generating the results..."):
        dataset_instance: DataInstance = st.session_state["dataset"][
            st.session_state.get("dataset_instance_id", 0)
        ]

        # Generate the NER results
        st.session_state["ner_results"] = st.session_state["ner"].convert_to_streamlit_output(
            dataset_instance.raw_text,
            st.session_state["ner"].extract_entities([dataset_instance.raw_text])[0],
        )

        st.session_state["dataset_instance"] = dataset_instance


def load_models():
    """Load the Models."""
    with st.spinner("Loading the NER model"):
        if st.session_state["ner_model"] == "regex":
            st.session_state["ner"] = RegexNer(weights_path=config.NER_REGEX_WEIGHTS_FILE)
        elif st.session_state["ner_model"] == "medcat":
            st.session_state["ner"] = MedCATNer()
        else:
            st.warning("No model selected")

    update_document()
