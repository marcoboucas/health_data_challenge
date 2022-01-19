"""Random Relation Extractor."""

from collections import defaultdict
from random import choice
from typing import Dict, List

from src.base.base_relextractor import BaseRelExtractor
from src.types import (
    EntityAnnotation,
    EntityAnnotationForRelation,
    RelationAnnotation,
    RelationValue,
)


class RandomRelExtractor(BaseRelExtractor):
    """Random model for evaluation."""

    def find_relations(
        self, texts: List[str], entities: List[List[EntityAnnotation]]
    ) -> List[List[RelationAnnotation]]:
        """Assess entities.

        :param texts: A list of texts
        :param entities: A list of entities per text
        """
        return [self.__find_relation_one(text, entities) for text, entities in zip(texts, entities)]

    # pylint: disable=unused-argument
    def __find_relation_one(self, text: str, entities: List[EntityAnnotation]):
        """Find one relation type."""
        relations = []
        entities_per_line: Dict[int, List[EntityAnnotation]] = defaultdict(list)
        for entity in entities:
            entities_per_line[entity.start_line].append(entity)

        for line_entities in entities_per_line.values():
            line_problems = list(filter(lambda x: x.label == "problem", line_entities))
            if len(line_entities) < 2 or len(line_problems) == 0:
                continue
            for problem in line_problems:
                for other_entity in line_entities:
                    if problem.start_word == other_entity.start_word:
                        continue

                    relations.append(
                        RelationAnnotation(
                            label=self.__random_relation(),
                            left_entity=EntityAnnotationForRelation(
                                **{k: v for k, v in other_entity.__dict__.items() if k != "label"}
                            ),
                            right_entity=EntityAnnotationForRelation(
                                **{k: v for k, v in problem.__dict__.items() if k != "label"}
                            ),
                        )
                    )

        relations = list(filter(lambda x: x != RelationValue.NO_RELATION, relations))
        return relations

    @staticmethod
    def __random_relation() -> str:
        """Return a random relation."""
        return choice(list(RelationValue)).value
