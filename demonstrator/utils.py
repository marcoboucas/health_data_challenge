"""Utils."""

import json
from collections import Counter
from random import choice, randint, seed
from typing import Dict, List

import numpy as np
import streamlit as st

PATIENT_NAMES = [
    "James Bond",
    "Hermione Granger",
    "Marc Bouckas",
    "Harry Potter",
    "Draco Malfoy",
    "Albus Perceval Wulfric Brian Dumbledore",
    "Severus Albus Potter",
    "Nicolas Laubé",
    "Captain Brownie",
    "Donald Trump",
    "Bruce Wayne",
    "Peter Parker",
    "Jack Napier",
    "Barack Obama",
    "Novak Djokovic",
    "Geralt of Rivia",
    "Tony Soprano",
    "Walter White",
    "Magnus Carlsen",
    "Daenerys Stormborn of House Targaryen, the First of Her Name, \
    Queen of the Andals and the First Men, Protector of the Seven \
    Kingdoms, the Mother of Dragons, the Khaleesi of \
    the Great Grass Sea, the Unburnt, the Breaker of Chains",
    "Tyrion Lannister",
    "Jon Snow... He is king of the North",
    "Elon Musk",
]

PROBLEMS = ["headache", "lower body pain", "dizzyness"]
TESTS = [
    "svr",
    "doppler",
    "palpation",
    "ldl",
    "lymphocytes",
    "trig",
    "mri",
    "totbili",
    "radiograph",
    "urean",
    "methgb",
    "bnp",
    "vma",
    "colonoscopy",
    "electrocardiogram",
]
TREATMENTS = [
    "immunization",
    "tetracyclines",
    "supplements",
    "dilatation",
    "marinol",
    "klonopin",
    "nitrates",
    "iron",
    "ortho-tri-cyclen",
    "sutures",
    "catheter",
    "sinemet",
    "sedated",
    "clomipramine",
    "clopidogrel",
    "rosuvastatin",
]


def get_cluster_name(patients: List[Dict]) -> str:
    """Get the cluster name from the patients."""
    counter = Counter()
    for patient in patients:
        counter.update(patient["problems"])
        counter.update(patient["tests"])
        counter.update(patient["treatments"])
    return ", ".join(list(map(lambda x: x[0], counter.most_common(3))))


def load_fake_data():
    """Loads fake data"""

    with open("./demonstrator/clusters.json", "r", encoding="utf-8") as file:
        data: dict = json.load(file)

    seed(666)
    for cluster in data.values():
        # all_labels = (
        #     cluster["problem_labels"] + cluster["tests_labels"] + cluster["treatment_labels"]
        # )

        cluster["patients"] = [generate_one_patient(cluster) for _ in range(randint(5, 20))]

        cluster["name"] = get_cluster_name(cluster["patients"])
    return data


@st.cache
def load_data(fake: bool = True):
    """Load the data."""
    if fake:
        return load_fake_data()
    return None


def get_username() -> str:
    """Return user names"""
    return choice(PATIENT_NAMES)


def generate_one_patient(cluster):
    """Generate one patient."""

    problems = list(
        set(list(np.random.choice(cluster["problem_labels"], size=randint(1, 2))))
        | set(list(np.random.choice(PROBLEMS, size=randint(0, 2))))
    )
    tests = list(
        set(list(np.random.choice(cluster["tests_labels"], size=randint(1, 2))))
        | set(list(np.random.choice(TESTS, size=randint(0, 2))))
    )
    treatments = list(
        set(list(np.random.choice(cluster["treatment_labels"], size=randint(1, 2))))
        | set(list(np.random.choice(TREATMENTS, size=randint(0, 2))))
    )
    return dict(
        name=get_username(),
        problems=problems,
        tests=tests,
        treatments=treatments,
    )
