"""Main types."""
# pylint: disable=invalid-name
from dataclasses import dataclass
from enum import Enum


class AssertionValue(Enum):
    """Value for the assertion class."""

    PRESENT = "present"
    ABSENT = "absent"
    POSSIBLE = "possible"
    CONDITIONAL = "conditional"
    HYPOTHETICAL = "hypothetical"
    NOT_PATIENT = "associated_with_someone_else"
    O_TOKEN = "O_TOKEN"


@dataclass
class Token:
    """One token (entity for NER)."""

    label: str
    text: str
    line: int
    word_index: int


@dataclass
class EntityAnnotation:
    """Entity Annotation"""

    label: str
    text: str
    start_line: int
    end_line: int
    start_word: int
    end_word: int


class RelationValue(Enum):
    """Relation values."""

    TrAP = "TrAP"
    TrNAP = "TrNAP"
    TrCP = "TrCP"
    TeRP = "TeRP"
    TeCP = "TeCP"
    TrIP = "TrIP"
    PIP = "PIP"
    TrWP = "TrWP"


@dataclass
class EntityAnnotationForRelation:
    """Entity annotation for relation."""

    text: str
    start_line: int
    end_line: int
    start_word: int
    end_word: int


@dataclass
class RelationAnnotation:
    """One relation."""

    label: RelationValue
    left_entity: EntityAnnotationForRelation
    right_entity: EntityAnnotationForRelation
