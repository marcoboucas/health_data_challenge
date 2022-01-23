"""Random Assessor."""

from copy import copy
from random import choice
from typing import List

from src.base.base_assessor import BaseAssessor
from src.types import AssertionValue, EntityAnnotation


class RandomAssessor(BaseAssessor):
    """Random Assessor."""

    def assess_entities(
        self, texts: List[str], concepts: List[List[EntityAnnotation]]
    ) -> List[List[EntityAnnotation]]:
        """Assess entities.

        :param texts: A list of texts
        :param entities: A list of entities per text
        """
        return [
            self.__assess_entities_one(text, entities) for text, entities in zip(texts, concepts)
        ]

    # pylint: disable=unused-argument
    def __assess_entities_one(
        self, text: str, entities: List[EntityAnnotation]
    ) -> List[EntityAnnotation]:
        """Assess entities for one text."""
        x = [copy(entity) for entity in entities]
        for entity in x:
            entity.label = self.__random_assertion()
        return x

    @staticmethod
    def __random_assertion() -> str:
        """Return a random assertion."""
        return choice(list(AssertionValue)).value
