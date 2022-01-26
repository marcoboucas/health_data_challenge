"""Pipeline"""
import os
from typing import List

import streamlit as st

from demonstrator.utils import get_username
from src.models import get_ner


class Pipeline:
    """Pipeline"""

    def __init__(
        self,
        data_folder: str = "./data/val/txt/",
        ner: str = "regex",
        assertion_ner: str = "random_ner",
        relation_ner: str = "realtion_ner",
    ) -> None:

        self.data_folder = data_folder
        self.num_clusters = 3
        self.device = "cpu"

        self.ner = ner
        self.assertion_bert = assertion_ner
        self.relation_ner = relation_ner

    def __load_files(self) -> List[str]:
        """Loads all files of folder"""
        texts = []
        files = os.listdir(self.data_folder)
        for file in files:
            if os.path.isfile(os.path.join(self.data_folder, file)):
                with open(os.path.join(self.data_folder, file), "r", encoding="utf-8") as file:
                    texts.append(" ".join(file.readlines()))
        return texts

    def run(self):
        """Runs the pipeline"""
        texts = self.__load_files()[:3]
        st.write(f"Running on {len(texts)}")
        ner = get_ner(self.ner)
        patients_concepts = ner.extract_entities(texts)
        del ner

        # assertions: List[List[EntityAnnotation]] = self.assertion_bert.assess_entities(
        #     texts, concepts
        # )

        patients = []
        print(patients_concepts)
        for patient_concepts in patients_concepts:
            patient = dict(name=get_username(), test=[], problem=[], treatment=[])

            for concept in patient_concepts:
                patient[concept.label].append(concept.text)

            print(patient)

        return patients

    @staticmethod
    def generate_clusters(patients: List[str]):
        """Generate clusters from a list of patients."""
        clusters = {}
        batch_size = 5

        for i in range(0, len(patients), batch_size):
            cluster = {}
            cluster["patients"] = patients[i : i + batch_size]
            # cluster["name"] = get_cluster_name(cluster["patients"])

            # for x in ["problem", "l"]:
            #     pass

        return clusters
