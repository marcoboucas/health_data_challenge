"""Base relation extractor model."""

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Tuple

from src.types import EntityAnnotation, RelationAnnotation, RelationValue


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
                if relation.label == RelationValue.NO_RELATION.value:
                    continue
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

    @staticmethod
    def find_interesting_lines_from_entities(
        text: str, entities: List[EntityAnnotation]
    ) -> List[Tuple[str, List[Tuple[EntityAnnotation, EntityAnnotation]]]]:
        """Find the interesting lines for relations."""
        interesting_lines = []
        lines = text.split("\n")

        entities_per_line: Dict[int, List[EntityAnnotation]] = defaultdict(list)
        for entity in entities:
            entities_per_line[entity.start_line].append(entity)

        for line_idx, line_entities in entities_per_line.items():
            line_problems = list(filter(lambda x: x.label == "problem", line_entities))
            if len(line_entities) < 2 or len(line_problems) == 0:
                continue

            # It's an interesting line !
            line = lines[line_idx - 1]
            entities_in_relation = []

            for problem in line_problems:
                for other_entity in line_entities:
                    if problem.start_word == other_entity.start_word:
                        continue
                    # Remove duplicates (because both problems)
                    if (
                        other_entity.label == "problem"
                        and other_entity.start_word < problem.start_word
                    ):
                        break

                    entities_in_relation.append((problem, other_entity))
            interesting_lines.append((line, entities_in_relation))
        return interesting_lines
