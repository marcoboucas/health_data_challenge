import streamlit as st

from src.pipeline.pipeline import Pipeline


@st.cache(suppress_st_warning=True)
def run_pipeline(data_folder):
    """Runs pipeline"""
    model_pipe = Pipeline(data_folder=data_folder)

    with st.spinner("Run the pipeline"):
        result = model_pipe.run()
    # st.session_state["pipeline_results"] = result

    return result


def uploader():
    """Uploader"""
    with st.form("Run the pipeline on custom data"):
        st.text_input("Upload folder", value="./data/val/txt/", key="data-folder")
        st.checkbox("Demonstration Mode", value=True, key="is-fake-data")
        if st.form_submit_button("Launch pipeline"):
            run_pipeline(st.session_state["data-folder"])

    # st.json(st.session_state.get("pipeline_results", "['not yet']"))
