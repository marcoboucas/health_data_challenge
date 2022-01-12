"""Base NER."""

import logging
from abc import ABC, abstractmethod
from typing import List

from src.types import EntityAnnotation

NER_LABELS = {"problem", "test", "treatment"}


class BaseNer(ABC):
    """Base Ner."""

    __name__ = "BaseNER"

    def __init__(self) -> None:
        """Init."""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Init the NER...")

    @abstractmethod
    def extract_entities(self, texts: List[str]) -> List[List[EntityAnnotation]]:
        """Extract entities from a list of texts.

        :param texts: A list of texts
        :returns: A list of tokens per text.
        """

    @staticmethod
    def entities_to_file(entities: List[EntityAnnotation], file_path: str) -> None:
        """Convert a list of entities to a file, using the conventionnal formating."""
        with open(file_path, "w", encoding="utf-8") as file:
            for entity in entities:
                file.write(
                    (
                        f'c="{entity.text}" {entity.start_line}:{entity.start_word} '
                        f"{entity.end_line}:{entity.end_word} "
                        f'||t="{entity.label}"\n'
                    )
                )
