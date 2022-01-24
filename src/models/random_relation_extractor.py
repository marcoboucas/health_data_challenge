"""Random Relation Extractor."""

from random import choice
from typing import List

from src.base.base_relation_extractor import BaseRelExtractor
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

        for _, entities_in_relation in self.find_interesting_lines_from_entities(text, entities):
            for ent1, ent2 in entities_in_relation:
                relations.append(
                    RelationAnnotation(
                        label=self.__random_relation(),
                        left_entity=EntityAnnotationForRelation(
                            **{k: v for k, v in ent1.__dict__.items() if k != "label"}
                        ),
                        right_entity=EntityAnnotationForRelation(
                            **{k: v for k, v in ent2.__dict__.items() if k != "label"}
                        ),
                    )
                )

        relations = list(filter(lambda x: x != RelationValue.NO_RELATION, relations))
        return relations

    @staticmethod
    def __random_relation() -> str:
        """Return a random relation."""
        return choice(list(RelationValue)).value
