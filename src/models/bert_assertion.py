"""Assertion BERT model"""
from pprint import pprint
from typing import List

import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.base.base_assessor import BaseAssessor
from src.types import EntityAnnotation


class BertAssertionnNER(BaseAssessor):
    """Bert Assertion NER"""

    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("bvanaken/clinical-assertion-negation-bert")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "bvanaken/clinical-assertion-negation-bert"
        )
        super().__init__()

    def assess_entities(
        self, texts: List[str], entities: List[List[EntityAnnotation]]
    ) -> List[List[EntityAnnotation]]:
        """Assertion prediction functions"""
        assertion_labels = []
        for text, entities_of_text in zip(texts, entities):
            text = self.format_text(text, entities_of_text).split("\n")
            tokenized_input = self.tokenizer(text, return_tensors="pt", padding=True)
            output = self.model(**tokenized_input)
            predicted_label = np.argmax(output.logits.detach().numpy(), axis=1)  ## 1 == ABSENT
            # print(predicted_label)
            assertion_labels.append(self.format_assess_entities(predicted_label, entities_of_text))

        return assertion_labels

    def format_assess_entities(
        self, labels: List[int], entities: List[EntityAnnotation]
    ) -> List[EntityAnnotation]:
        """Formats the entities"""
        # we suppose that both list are ordered
        return [
            EntityAnnotation(
                label=self.get_label(label),
                text=entity.text,
                start_line=entity.start_line,
                start_word=entity.start_word,
                end_line=entity.end_line,
                end_word=entity.end_word,
            )
            for label, entity in zip(labels, entities)
        ]

    @staticmethod
    def get_label(label: int) -> str:
        """Returns label as string
        PRESENT(0), ABSENT(1) or POSSIBLE(2)
        """
        if label == 0:
            return "present"
        if label == 1:
            return "absent"
        return "possible"

    @staticmethod
    def format_text(text: str, entities: List[EntityAnnotation]) -> str:
        """Returns a text formated for assertion prediction

        Ex: He probably has migraine. (migraine is a problem entity)
        returns: He probably has [entity].
        """
        tag_entity = "[entity]"
        text_list = [[word for word in line.split(" ") if word != ""] for line in text.split("\n")]
        for entity in entities:
            tag_added = False
            if entity.label == "problem":
                end_word = entity.end_word
                found = ""
                for line_id in range(entity.start_line, entity.end_line + 1):
                    if entity.start_line - entity.end_line == 0:
                        end_index = end_word
                    elif line_id == entity.end_line:
                        end_index = end_word
                    else:
                        end_index = len(text_list[entity.start_line - 1])

                    for word_id in range(entity.start_word, end_index + 1):
                        found += text_list[entity.start_line - 1][word_id] + " "
                        if not tag_added:
                            text_list[entity.start_line - 1][word_id] = tag_entity
                            tag_added = True
                        else:
                            text_list[entity.start_line - 1][word_id] = ""

        return "\n".join([" ".join(word) for word in text_list if word != ""])


if __name__ == "__main__":
    from src.dataset.dataset_loader import DatasetLoader

    dataset_loader = DatasetLoader(mode="train")
    berter_assert = BertAssertionnNER()

    for i, data in enumerate(dataset_loader):
        if i == 0:
            print(data.raw_text)
            pprint(berter_assert.assess_entities([data.raw_text], [data.annotation_concept]))
