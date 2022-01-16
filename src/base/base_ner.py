"""Base NER."""

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Tuple, Union

from src.types import EntityAnnotation

NER_LABELS = {"problem", "test", "treatment"}


class BaseNer(ABC):
    """Base Ner."""

    STREAMLIT_COLORS = {
        "problem": "#faa",
        "test": "#8ef",
        "treatment": "#fea",
    }

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
        assert file_path.endswith(".con"), "The file must be a .con file"
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
        text_length = 0
        for line_num, line in enumerate(text.split("\n")):
            if character_position < text_length + (len(line) + 1):
                line_length = 0
                for word_position, word in enumerate(line.split(" ")):
                    if character_position <= text_length + line_length + len(word):
                        return line_num + 1, word_position

                    line_length += len(word) + 1  # word length + ' '

            text_length += len(line) + 1  # line length + \n char

        return -1, -1

    def convert_to_streamlit_output(
        self, text: str, entities: List[EntityAnnotation]
    ) -> List[Union[str, Tuple[str, str, str]]]:
        """Convert to the streamlit output."""
        # Convert the entities line_idx, word_idx to their character_position
        entities_per_line = defaultdict(list)
        for entity in entities:
            entities_per_line[entity.start_line - 1].append(entity)

        final_output = []
        for line_num, line in enumerate(text.split("\n")):
            if line_num in entities_per_line:
                words = line.split(" ")
                word_idx = 0
                for entity in entities_per_line[line_num]:
                    final_output.append(" ".join(words[word_idx : entity.start_word]))
                    final_output.append(
                        (entity.text, entity.label, self.STREAMLIT_COLORS[entity.label])
                    )
                    word_idx = entity.end_word + 1
                if word_idx < len(words):
                    final_output.append(words[word_idx:])
            else:
                final_output.append(line)
            final_output.append(" \n ")

        return final_output
