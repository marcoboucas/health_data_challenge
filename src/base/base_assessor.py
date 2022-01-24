"""Base Assessor."""

import logging
from abc import ABC, abstractmethod
from typing import List

from src.types import EntityAnnotation


class BaseAssessor(ABC):
    """Base Assessor."""

    def __init__(self) -> None:
        """Init."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Init the Assessor...")

    @abstractmethod
    def assess_entities(
        self, texts: List[str], concepts: List[List[EntityAnnotation]]
    ) -> List[List[EntityAnnotation]]:
        """Assess entities.

        :param texts: A list of texts
        :param entities: A list of entities per text
        """

    @staticmethod
    def assertions_to_file(assertions: List[EntityAnnotation], file_path: str) -> None:
        """Save assertions into a file, using the same convention."""
        assert file_path.endswith(".ast"), "The file must be a .ast file"
        with open(file_path, "w", encoding="utf-8") as file:
            for assertion in assertions:
                escaped_text = assertion.text.replace("\n", " ")
                file.write(
                    (
                        f'c="{escaped_text}" {assertion.start_line}:{assertion.start_word} '
                        f"{assertion.end_line}:{assertion.end_word}"
                        '||t="problem"'
                        f'||a="{assertion.label}"\n'
                    )
                )
