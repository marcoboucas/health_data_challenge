"""Pipeline"""
import os
from collections import defaultdict
from random import choice
from typing import List

import streamlit as st

from demonstrator.utils import get_cluster_name, get_username
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
        texts = self.__load_files()
        st.write(f"Running on {len(texts)}")
        ner = get_ner(self.ner)
        patients_concepts = ner.extract_entities(texts)
        del ner

        patients = []
        for patient_concepts in patients_concepts:
            patient_information = defaultdict(list)
            patient_pbs = defaultdict(dict)
            for concept in patient_concepts:
                if concept.label != "problem":
                    patient_information[concept.label + "s"].append(concept.text)
                else:
                    patient_pbs[concept.label + "s"][concept.text] = self.get_assertion()
            result = {
                "name": get_username(),
            }
            for key in patient_information:
                result[key] = patient_information[key]
            for key in patient_pbs:
                result[key] = patient_pbs[key]
            patients.append(result)

        return self.generate_clusters(patients)

    @staticmethod
    def get_assertion():
        """Get assertion"""
        return choice(["present", "absent", "hypothetical"])

    @staticmethod
    def generate_clusters(patients: List[str]):
        """Generate clusters from a list of patients."""
        clusters = {}
        batch_size = 5

        for i in range(0, len(patients), batch_size):
            cluster = defaultdict(list)
            selected_patients = patients[i : i + batch_size]
            cluster["patients"] = selected_patients

            cluster["name"] = get_cluster_name(selected_patients)

            for patient in selected_patients:
                for key in patient:
                    if key not in ["name", "problem"]:
                        cluster[key + "_labels"].extend(patient[key])
                    elif key == "problem":
                        cluster[key + "_labels"].extend(patient[key].keys())
            for key in cluster:
                if key not in ["name", "patients"]:
                    cluster[key] = list(set(cluster[key]))

            clusters[i] = cluster
        return clusters
