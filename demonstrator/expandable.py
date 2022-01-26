# pylint: disable=C0411,E0401
from collections import Counter

import pandas as pd
import plotly.express as px
import streamlit as st


def expandable_cluster(cluster):
    """Expandable cluster"""

    with st.expander(f"{cluster['name']}... ({len(cluster['patients'])} persons)"):
        columns = ["name", "problems", "treatments", "tests"]

        df = pd.DataFrame(
            [
                [
                    patient["name"],
                    ", ".join(patient["problems"]),
                    ", ".join(patient["treatments"]),
                    ", ".join(patient["tests"]),
                ]
                for patient in cluster["patients"]
            ],
            columns=columns,
        )

        st.dataframe(df)

        with st.container():
            cols = st.columns(3)
            for col, entity_type in zip(cols, ["problems", "treatments", "tests"]):
                counter = Counter()
                for patient in cluster["patients"]:
                    counter.update(patient[entity_type])
                fig = px.bar(
                    pd.DataFrame(counter.most_common(), columns=[entity_type, "count"]).head(5),
                    x=entity_type,
                    y="count",
                    title=f"Distribution of {entity_type}",
                )
                with col:
                    st.plotly_chart(fig, use_container_width=True)
