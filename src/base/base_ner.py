"""Base NER."""

import logging
from abc import ABC, abstractmethod
from typing import List, Tuple

from src.types import EntityAnnotation

NER_LABELS = {"problem", "test", "treatment"}


class BaseNer(ABC):
    """Base Ner."""

    def __init__(self) -> None:
        """Init."""
        self.logger = logging.getLogger(self.__class__.__name__)
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
                escaped_text = entity.text.replace("\n", " ")
                file.write(
                    (
                        f'c="{escaped_text}" {entity.start_line}:{entity.start_word} '
                        f"{entity.end_line}:{entity.end_word} "
                        f'||t="{entity.label}"\n'
                    )
                )

    @staticmethod
    def character_to_line_and_word(text: str, character_position: int) -> Tuple[int, int]:
        """Return line and word position from index.

        - The line (starts at 1)
        - The word position (starts at 0)
        """
        lines = text.split("\n")

        text_length = 0
        for line_num, line in enumerate(lines):
            if character_position < text_length + (len(line) + 1):
                line_length = 0
                for word_position, word in enumerate(line.split(" ")):
                    if character_position <= text_length + line_length + len(word):
                        return line_num + 1, word_position

                    line_length += len(word) + 1  # word length + ' '

            text_length += len(line) + 1  # line length + \n char

        return -1, -1
