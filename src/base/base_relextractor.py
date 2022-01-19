"""Base relation extractor model."""

import logging
from abc import ABC, abstractmethod
from typing import List

from src.types import EntityAnnotation, RelationAnnotation


class BaseRelExtractor(ABC):
    """Base RelExtractor."""

    def __init__(self) -> None:
        """Init."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Init the NER...")

    @abstractmethod
    def find_relations(
        self, texts: List[str], entities: List[List[EntityAnnotation]]
    ) -> List[List[RelationAnnotation]]:
        """Assess entities.

        :param texts: A list of texts
        :param entities: A list of entities per text
        """

    @staticmethod
    def relations_to_file(relations: List[RelationAnnotation], file_path: str) -> None:
        """Convert a list of relations to a file, using the conventional format."""
        assert file_path.endswith(".rel"), "The file must be a .rel file"
        with open(file_path, "w", encoding="utf-8") as file:
            for relation in relations:
                entity_left_text = relation.left_entity.text.replace("\n", " ")
                entity_right_text = relation.left_entity.text.replace("\n", " ")
                file.write(
                    (
                        f'c="{entity_left_text}" '
                        f"{relation.left_entity.start_line}:{relation.left_entity.start_word} "
                        f"{relation.left_entity.end_line}:{relation.left_entity.end_word}"
                        f'||r="{relation.label}"||'
                        f'c="{entity_right_text}"'
                        f"{relation.right_entity.start_line}:{relation.right_entity.start_word} "
                        f"{relation.right_entity.end_line}:{relation.right_entity.end_word}\n"
                    )
                )
