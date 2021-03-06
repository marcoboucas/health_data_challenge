# Health AI challenge

[![wakatime](https://wakatime.com/badge/github/marcoboucas/health_data_challenge.svg)](https://wakatime.com/badge/github/marcoboucas/health_data_challenge)
[![GitHub issues](https://img.shields.io/github/issues/marcoboucas/health_data_challenge)](https://github.com/marcoboucas/health_data_challenge/issues)
![GitHub branch checks state](https://img.shields.io/github/checks-status/marcoboucas/health_data_challenge/main)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/marcoboucas/health_data_challenge)

Health AI challenge organized by CentraleSupelec and ILLUIN Technology.
The script eval.py allows you to evaluate your approach to the first part of the health AI challenge.

## Installation

### Prerequisites

- python 3.8
- pip
- Knowledge of one of python virtual environments

### Package installation

- Install the dependencies: `pip install -r dev.requirements.txt`
- Prepare the data by running the command: `python -m src.dataset.extract` (You need to install the data from teams: https://centralesupelec.sharepoint.com/:f:/r/sites/Essai423-Sant_01/Documents%20partages/Sant%C3%A9_01/data?csf=1&web=1&e=ULYeMd)

- For MedSpacy, you need to install some weights here: https://medcat.rosalind.kcl.ac.uk/media/medmen_wstatus_2021_oct.zip, you can put them in the `./weights` folder.

## Training models

### NER

- For the Regex NER, you can do `python -m src.models.regex_ner`
- For the Medcat NER, you can do `python -m src.models.medcat_ner`
- For the Bert NER, yoi can do `python -m src.models.bert_ner`

### Access to data

Ask to your coach.

## Run evaluation script

```
python eval.py evaluate  \
--concept_annotation_dir=<path_to_concept_annotation_dir> \
--concept_prediction_dir=<path_to_concept_prediction_dir> \
--assertion_annotation_dir=<path_to_assertion_annotation_dir> \
--assertion_prediction_dir=<path_to_assertion_prediction_dir> \
--relation_annotation_dir=<path_to_relation_annotation_dir> \
--relation_prediction_dir=<path_to_relation_prediction_dir> \
--entries_dir=<path_to_medical_reports>
```
