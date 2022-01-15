"""Regex NER."""

import os
import pickle
import re
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Set, Union

import dacite

from src import config
from src.base.base_ner import NER_LABELS, BaseNer
from src.types import EntityAnnotation, Token


@dataclass
class RegexNerWeights:
    """Weights for the regex NER."""

    problem: Set[str] = field(default_factory=set)
    treatment: Set[str] = field(default_factory=set)
    test: Set[str] = field(default_factory=set)

    pattern: Optional[re.Pattern] = None


class RegexNer(BaseNer):
    """Regex NER."""

    def __init__(self, weights_path: Optional[str] = None) -> None:
        """Init."""
        super().__init__()
        self.weights = RegexNerWeights()
        if weights_path is not None:
            self.logger.info("Loading the pretrained weights")
            self.load_from_weights(weights_path=weights_path)

    def extract_entities(self, texts: List[str]) -> List[List[EntityAnnotation]]:
        """Extract entities from a list of texts."""
        return list(map(self.__extract_entities_one_file, texts))

    def __extract_entities_one_file(self, text: str) -> List[EntityAnnotation]:
        """Extract entities for one file."""
        tokens = []
        # We suppose an entity can be found only on one line
        for match in self.weights.pattern.finditer(text):
            label = self.find_token_label(match.group(0))
            start_line, start_word = self.character_to_line_and_word(text, match.start())
            end_line, end_word = self.character_to_line_and_word(text, match.end())
            if label in NER_LABELS:
                tokens.append(
                    EntityAnnotation(
                        label=label,
                        text=match.group(0),
                        start_line=start_line,
                        end_line=end_line,
                        start_word=start_word,
                        end_word=end_word + 1,
                    )
                )
        return tokens

    def load_from_weights(self, weights_path: str = config.NER_REGEX_WEIGHTS_FILE) -> None:
        """Load the weights."""
        if not os.path.isfile(weights_path):
            self.logger.warning("No file found here: '%s'", weights_path)
            return
        with open(weights_path, "rb") as file:
            self.weights = dacite.from_dict(RegexNerWeights, pickle.load(file))

    def save_weights(self, weights_path: str = config.NER_REGEX_WEIGHTS_FILE) -> None:
        """Save the weights in a file."""
        with open(weights_path, "wb") as file:
            pickle.dump(asdict(self.weights), file)

    def train(self, annotations: List[List[Union[Token, EntityAnnotation]]]) -> None:
        """Train the model."""
        for annotation in annotations:
            for token in annotation:
                if token.label in NER_LABELS:
                    if len(token.text.strip()) > 2:
                        getattr(self.weights, token.label).add(token.text.lower())
        # Generate the patterns
        self.weights.pattern = re.compile(
            r"("
            + r"|".join(
                list(
                    map(
                        self.escape_characters,
                        self.weights.test | self.weights.problem | self.weights.treatment,
                    )
                )
            )
            + r")",
            flags=re.IGNORECASE,
        )

    def find_token_label(self, token: str) -> str:
        """Find token label."""
        x = token.lower()
        for label in NER_LABELS:
            if x in getattr(self.weights, label):
                return label
        return "no_label"

    @staticmethod
    def escape_characters(text: str) -> str:
        """Escape characters."""
        return re.escape(text).strip()


if __name__ == "__main__":
    import logging
    from pprint import pprint

    from src.dataset.dataset_loader import DatasetLoader

    logging.basicConfig(level=logging.INFO)
    ner = RegexNer()
    logging.info("Training the model !")
    train_set = DatasetLoader(mode="train")
    ner.train([train_set[i].annotation_concept for i in range(len(train_set))])
    logging.info("Training done !")
    pprint(ner.extract_entities(["I have pain in my lower body"]))
    ner.save_weights()
