import logging
import os
import re
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Union

from src.types import (
    EntityAnnotation,
    EntityAnnotationForRelation,
    RelationAnnotation,
    RelationValue,
)


class Parser:
    """Document parser."""

    def __init__(self) -> None:
        pass

    @staticmethod
    def parse_annotation_relation(path: str) -> List[RelationAnnotation]:
        """Parse one relation file."""
        return parser(parse_one_relation, path)

    @staticmethod
    def parse_annotation_concept(path: str) -> List[EntityAnnotation]:
        """Parse one concept file."""
        return parser(parse_one_concept, path)

    @staticmethod
    def parse_annotation_assertion(path: str) -> List[EntityAnnotation]:
        """Parse one assertion file."""
        return parser(parse_one_assertion, path)

    @staticmethod
    def parse_raw_text(raw_text_path: str) -> Dict[str, str]:
        """Parse the raw text."""

        formated_text = defaultdict(str)

        with open(raw_text_path, "r", encoding="utf-8") as raw:
            lines = raw.readlines()
            current_key = clean_section_name(lines[0])
            for line in lines:
                if is_section_title(line):
                    current_key = clean_section_name(line)
                else:
                    formated_text[current_key] += clean_txt(line)

        return dict(formated_text)

    @staticmethod
    def get_raw_text(raw_text_path: str) -> List[str]:
        """Return the raw text."""
        with open(raw_text_path, "r", encoding="utf-8") as raw:
            lines = raw.readlines()
        return lines


def is_section_title(line: str) -> bool:
    """Check if a line is a title (key of dictionary)."""
    # double points at end
    if re.search(r":$", line):
        return True
    # only capital
    if line.isupper():
        return True
    return False


def clean_section_name(section_name: str) -> str:
    """Clean a name section."""
    return section_name.replace("\n", "").replace(" :", "").replace(":", "")


def clean_txt(line: str) -> str:
    """Clean a line by removing unwanted characters."""
    return line.replace("_", "").replace("\n", "")


def parser(func: Callable, path: str) -> Union[List[RelationAnnotation], List[EntityAnnotation]]:
    """Parse one file, either relation, concept or assertion."""
    entities_annotated = []
    with open(os.path.abspath(path), "r", encoding="utf-8") as input_file:
        for entity_line in input_file.readlines():
            entity_annotation = func(entity_line)
            if entity_annotation is not None:
                entities_annotated.append(func(entity_line))
            else:
                logging.warning("Failed to parse annotation %s for file %s", entity_line, path)
    return entities_annotated


def parse_one_relation(line: str) -> Optional[EntityAnnotation]:
    """Parse one relation file."""
    try:
        return RelationAnnotation(
            label=RelationValue[
                line.split("||")[1].split("=")[1].replace('"', "").replace("\n", "")
            ],
            left_entity=EntityAnnotationForRelation(
                text=re.split(r"(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})", line.split("||")[0])[0]
                .split("=")[1]
                .replace('"', ""),
                start_line=int(
                    re.split(r"(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})", line.split("||")[0])[1]
                    .split(" ")[0]
                    .split(":")[0]
                ),
                start_word=int(
                    re.split(r"(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})", line.split("||")[0])[1]
                    .split(" ")[0]
                    .split(":")[1]
                ),
                end_line=int(
                    re.split(r"(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})", line.split("||")[0])[1]
                    .split(" ")[1]
                    .split(":")[0]
                ),
                end_word=int(
                    re.split(r"(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})", line.split("||")[0])[1]
                    .split(" ")[1]
                    .split(":")[1]
                ),
            ),
            right_entity=EntityAnnotationForRelation(
                text=re.split(r"(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})", line.split("||")[2])[0]
                .split("=")[1]
                .replace('"', ""),
                start_line=int(
                    re.split(r"(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})", line.split("||")[2])[1]
                    .split(" ")[0]
                    .split(":")[0]
                ),
                start_word=int(
                    re.split(r"(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})", line.split("||")[2])[1]
                    .split(" ")[0]
                    .split(":")[1]
                ),
                end_line=int(
                    re.split(r"(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})", line.split("||")[2])[1]
                    .split(" ")[1]
                    .split(":")[0]
                ),
                end_word=int(
                    re.split(r"(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})", line.split("||")[2])[1]
                    .split(" ")[1]
                    .split(":")[1]
                ),
            ),
        )
    except (ValueError, IndexError):
        return None


def parse_one_concept(text: str) -> Optional[EntityAnnotation]:
    """Parse one concept file."""
    try:
        return EntityAnnotation(
            label=text.split("||")[1].split("=")[1].replace('"', "").replace("\n", ""),
            text=re.split(r"(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})", text.split("||")[0])[0]
            .split("=")[1]
            .replace('"', ""),
            start_line=int(
                re.split(r"(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})", text.split("||")[0])[1]
                .split(" ")[0]
                .split(":")[0]
            ),
            start_word=int(
                re.split(r"(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})", text.split("||")[0])[1]
                .split(" ")[0]
                .split(":")[1]
            ),
            end_line=int(
                re.split(r"(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})", text.split("||")[0])[1]
                .split(" ")[1]
                .split(":")[0]
            ),
            end_word=int(
                re.split(r"(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})", text.split("||")[0])[1]
                .split(" ")[1]
                .split(":")[1]
            ),
        )
    except (ValueError, IndexError):
        return None


def parse_one_assertion(text: str) -> Optional[EntityAnnotation]:
    """Parse one assertion file."""
    try:
        return EntityAnnotation(
            label=text.split("||")[2].split("=")[1].replace('"', "").replace("\n", ""),
            text=re.split(r"(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})", text.split("||")[0])[0]
            .split("=")[1]
            .replace('"', ""),
            start_line=int(
                re.split(r"(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})", text.split("||")[0])[1]
                .split(" ")[0]
                .split(":")[0]
            ),
            start_word=int(
                re.split(r"(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})", text.split("||")[0])[1]
                .split(" ")[0]
                .split(":")[1]
            ),
            end_line=int(
                re.split(r"(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})", text.split("||")[0])[1]
                .split(" ")[1]
                .split(":")[0]
            ),
            end_word=int(
                re.split(r"(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})", text.split("||")[0])[1]
                .split(" ")[1]
                .split(":")[1]
            ),
        )
    except (ValueError, IndexError):
        return None


if __name__ == "__main__":
    text_file = os.path.join("data", "train_data", "partners", "txt", "018636330_DH.txt")
