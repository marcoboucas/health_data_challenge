"""Regex NER."""

import os
import pickle
import re
from dataclasses import dataclass, field
from typing import List, Optional, Set, Union

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
            self.load_from_weights(weights_path=weights_path)

    def extract_entities(self, texts: List[str]) -> List[List[EntityAnnotation]]:
        """Extract entities from a list of texts."""
        return list(map(self.__extract_entities_one_file, texts))

    def __extract_entities_one_file(self, text: str) -> List[EntityAnnotation]:
        """Extract entities for one file."""
        tokens = []
        # We suppose an entity can be found only on one line
        for i, line in enumerate(text.split("\n")):
            for match in self.weights.pattern.finditer(line):
                label = self.find_token_label(match.group(0))
                if label in NER_LABELS:
                    tokens.append(
                        EntityAnnotation(
                            label=label,
                            text=match.group(0),
                            start_line=i,
                            end_line=i,
                            start_word=match.start(),
                            end_word=match.end(),
                        )
                    )
        return tokens

    def load_from_weights(self, weights_path: str) -> None:
        """Load the weights."""
        if not os.path.isfile(weights_path):
            self.logger.warning("No file found here: '%s'", weights_path)
            return
        with open(weights_path, "rb") as file:
            self.weights = pickle.load(file)

    def save_weights(self, weights_path: str) -> None:
        """Save the weights in a file."""
        with open(weights_path, "wb") as file:
            pickle.dump(self.weights, file)

    def train(self, annotations: List[List[Union[Token, EntityAnnotation]]]) -> None:
        """Train the model."""
        for annotation in annotations:
            for token in annotation:
                if token.label in NER_LABELS:
                    getattr(self.weights, token.label).add(token.text.lower())
        # Generate the patterns
        self.weights.pattern = re.compile(
            r"("
            + r"|".join(self.weights.test | self.weights.problem | self.weights.treatment)
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


if __name__ == "__main__":
    import logging
    from pprint import pprint

    logging.basicConfig(level=logging.INFO)
    ner = RegexNer()

    ner.train([[Token("test", "electrocardiogram", 10, 1), Token("problem", "cough", 13, 1)]])

    pprint(
        ner.extract_entities(["I had an electrocardiogram\n and bad cough", "I have a tough cough"])
    )
