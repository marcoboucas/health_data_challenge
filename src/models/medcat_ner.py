"""MedCAT NER."""

import operator
import os
import pickle
from collections import defaultdict
from typing import Dict, List, Optional, Union

from medcat.cat import CAT

from src import config
from src.base.base_ner import BaseNer
from src.types import EntityAnnotation, Token


class MedCATNer(BaseNer):
    """NER using MedCat NER."""

    def __init__(
        self,
        model_path: Optional[str] = config.MEDCAT_ZIP_FILE,
        weights_path: Optional[str] = config.NER_MEDCAT_WEIGHTS_FILE,
    ) -> None:
        """Init."""
        super().__init__()
        self.cat = CAT.load_model_pack(model_path)  # type: ignore
        # Convert the label of medcat to a 'test', 'problem', 'treatment'
        self.label_mapping: Dict[str, str] = {}
        self.load_from_weights(weights_path)

    def extract_entities(self, texts: List[str]) -> List[List[EntityAnnotation]]:
        """Extract entities from a list of texts."""
        total_entities = []
        for text in texts:
            current_entities = []
            entities = self.cat.get_entities(text)
            for elem in entities["entities"].values():
                label = self.label_mapping.get(elem["types"][0])
                if not label:  # Invalid label
                    continue
                name = elem["source_value"]
                start_line, start_word = self.character_to_line_and_word(text, elem["start"])
                end_line, end_word = self.character_to_line_and_word(text, elem["end"])

                current_entities.append(
                    EntityAnnotation(
                        label=label,
                        text=name,
                        start_line=start_line,
                        end_line=end_line,
                        start_word=start_word,
                        end_word=end_word,
                    )
                )

            total_entities.append(current_entities)

        return total_entities

    def train(self, annotations: List[List[Union[Token, EntityAnnotation]]]) -> None:
        """Train the model."""
        labels_mapping_count = defaultdict(lambda: defaultdict(lambda: 0))
        for annotation in annotations:
            for token in annotation:
                entities = self.cat.get_entities(token.text)["entities"].values()
                for elem in entities:
                    for type_ in elem["types"]:
                        labels_mapping_count[type_][token.label] += 1
        self.logger.info("Training completed... Converting the label mapping to dict")
        # For each NER label, find the relevant tag based on the count
        self.label_mapping = {
            key: list(sorted(value.items(), key=operator.itemgetter(1), reverse=True))[0][0]
            for key, value in labels_mapping_count.items()
        }

    def save_weights(self, weights_path: str = config.NER_MEDCAT_WEIGHTS_FILE) -> None:
        """Save the weights in a file."""
        with open(weights_path, "wb") as file:
            pickle.dump(self.label_mapping, file)

    def load_from_weights(self, weights_path: str = config.NER_MEDCAT_WEIGHTS_FILE) -> None:
        """Load the weights."""
        if not os.path.isfile(weights_path):
            self.logger.warning("Can't find the weights: '%s'", weights_path)
            return
        with open(weights_path, "rb") as file:
            self.label_mapping = pickle.load(file)


if __name__ == "__main__":
    # pylint: disable=ungrouped-imports
    from tqdm import tqdm

    from src.dataset.dataset_loader import DatasetLoader

    dataset = DatasetLoader("train")
    ner = MedCATNer()
    ner.train([dataset[i].annotation_concept for i in tqdm(range(len(dataset)))])
    ner.save_weights()
