"""Document view."""

import streamlit as st

from demonstrator.utils import display_annotations


def document_view() -> None:
    """Document view."""
    with st.container():
        # st.write(st.session_state["dataset_instance"].raw_text.replace("\n", "  \n"))
        display_annotations(st.session_state["ner_results"])
