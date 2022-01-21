"""Init."""
# pylint: disable=import-outside-toplevel
# Important to avoid loading all the classes
# when importing the module
# It speeds up the code

from src import config
from src.base.base_assessor import BaseAssessor
from src.base.base_ner import BaseNer


def get_ner(ner_name: str, ner_path: str = None) -> BaseNer:
    """Get the requested NER model."""
    ner: BaseNer
    if ner_name == "regex":
        from src.models.regex_ner import RegexNer

        ner = RegexNer(weights_path=ner_path or config.NER_REGEX_WEIGHTS_FILE)
    elif ner_name == "medcat":
        from src.models.medcat_ner import MedCATNer

        ner = MedCATNer(weights_path=ner_path or config.NER_MEDCAT_WEIGHTS_FILE)
    else:
        raise ValueError(f"No '{ner_name}' NER model")
    return ner


def get_assessor(assessor_name: str) -> BaseAssessor:
    """Get the requested assessor."""
    assessor: BaseAssessor
    if assessor_name == "random":
        from src.models.random_assessor import RandomAssessor

        assessor = RandomAssessor()
    elif assessor_name == "bert":
        from src.models.bert_assertion import BertAssessor

        assessor = BertAssessor()
    else:
        raise ValueError(f"No '{assessor_name}' Assessor model")
    return assessor
