"""Init."""
# pylint: disable=import-outside-toplevel
# Important to avoid loading all the classes
# when importing the module
# It speeds up the code
from typing import Optional

from src import config
from src.base.base_assessor import BaseAssessor
from src.base.base_ner import BaseNer
from src.base.base_relation_extractor import BaseRelExtractor


def get_ner(ner_name: str, ner_path: str = None) -> BaseNer:
    """Get the requested NER model."""
    ner: BaseNer
    if ner_name == "regex":
        from src.models.regex_ner import RegexNer

        ner = RegexNer(weights_path=ner_path or config.NER_REGEX_WEIGHTS_FILE)
    elif ner_name == "medcat":
        from src.models.medcat_ner import MedCATNer

        ner = MedCATNer(weights_path=ner_path or config.NER_MEDCAT_WEIGHTS_FILE)
    elif ner_name == "bert":
        from src.models.bert_ner import BertNer

        ner = BertNer()
    else:
        raise ValueError(f"No '{ner_name}' NER model")
    return ner


# pylint: disable=unused-argument
def get_assessor(assessor_name: str, assessor_path: Optional[str] = None) -> BaseAssessor:
    """Get the requested assessor."""
    assessor: BaseAssessor
    if assessor_name == "random":
        from src.models.random_assessor import RandomAssessor

        assessor = RandomAssessor()
    elif assessor_name == "bert":
        from src.models.bert_assertion_on_sentences import BertAssessorSentences

        if assessor_path is not None:
            assessor = BertAssessorSentences(model_name=assessor_path)
        else:
            assessor = BertAssessorSentences()
    else:
        raise ValueError(f"No '{assessor_name}' Assessor model")
    return assessor


def get_relation_extractor(extractor_name: str, weights_path: str = None) -> BaseRelExtractor:
    """Get the requested relation extractor."""
    extractor: BaseRelExtractor

    if extractor_name == "random":
        from src.models.random_relation_extractor import RandomRelExtractor

        extractor = RandomRelExtractor()
    elif extractor_name == "huggingface":
        from src.models.bert_relation_extractor import BertRelExtractor

        extractor = BertRelExtractor(weights_path=weights_path)
    else:
        raise ValueError(f"No '{extractor_name}' Relation Extractor model")
    return extractor
