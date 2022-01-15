# Health AI challenge

Health AI challenge organized by CentraleSupelec and ILLUIN Technology.
The script eval.py allows you to evaluate your approach to the first part of the health AI challenge.

## Installation

### Prerequisites
* python 3.7.6
* pip
* Knowledge of one of python virtual environments

### Package installation

```bash
pip install -r dev.requirements.txt
```

- For MedSpacy, you need to install some weights here: https://medcat.rosalind.kcl.ac.uk/media/medmen_wstatus_2021_oct.zip, you can put them in the `./weights` folder.

## Training models
### NER
- For the Regex NER, you can do `python -m src.models.regex_ner`
- For the Medcat NER, you can do `python -m src.models.medcat_ner`
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
