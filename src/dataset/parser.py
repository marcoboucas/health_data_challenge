from typing import Dict, Optional, List, Union
from collections import defaultdict
import re
import os
import logging

from src.types import (
    EntityAnnotation,
    RelationAnnotation,
    RelationValue,
    EntityAnnotationForRelation,
)


class Parser:
    def __init__(self) -> None:
        pass

    def parse_annotation_relation(self, path: str) -> List[RelationAnnotation]:
        return parser(parse_one_relation, path)

    def parse_annotation_concept(self, path: str) -> List[RelationAnnotation]:
        return parser(parse_one_concept, path)

    def parse_annotation_assertion(self, path: str) -> List[RelationAnnotation]:
        return parser(parse_one_assertion, path)

    def parse_raw_text(self, raw_text_path: str) -> Dict[str, str]:

        formated_text = defaultdict(str)

        with open(raw_text_path, "r", encoding="utf-8") as raw:
            lines = raw.readlines()
            current_key = clean_section_name(lines[0])
            for line in lines:
                if is_section_title(line):
                    current_key = clean_section_name(line)
                else:
                    formated_text[current_key] += clean_txt(line)

        return formated_text


def is_section_title(line: str) -> bool:
    # double points at end
    if re.search(r":$", line):
        return True
    # only capital
    if line.isupper():
        return True
    return False


def clean_section_name(section_name: str) -> str:
    return section_name.replace("\n", "").replace(" :", "").replace(":", "")


def clean_txt(line: str) -> str:
    return line.replace("_", "").replace("\n", "")


def parser(func: callable, path: str) -> Union[List[RelationAnnotation], List[EntityAnnotation]]:
    entities_annotated = []
    entity_annotation = None
    with open(os.path.abspath(path), "r", encoding="utf-8") as input_file:
        for entity_line in input_file.readlines():
            if entity_annotation is not None:
                entities_annotated.append(func(entity_line))
            else:
                logging.warning("Failed to parse annotation %s for file %s", entity_line, path)
    return entities_annotated


def parse_one_relation(line: str) -> Optional[EntityAnnotation]:
    try:
        return RelationAnnotation(
            label=RelationValue[
                line.split("||")[1].split("=")[1].replace('"', "").replace("\n", "")
            ],
            left_entity=EntityAnnotationForRelation(
                text=re.split("(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})", line.split("||")[0])[0]
                .split("=")[1]
                .replace('"', ""),
                start_line=int(
                    re.split("(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})", line.split("||")[0])[1]
                    .split(" ")[0]
                    .split(":")[0]
                ),
                start_word=int(
                    re.split("(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})", line.split("||")[0])[1]
                    .split(" ")[0]
                    .split(":")[1]
                ),
                end_line=int(
                    re.split("(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})", line.split("||")[0])[1]
                    .split(" ")[1]
                    .split(":")[0]
                ),
                end_word=int(
                    re.split("(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})", line.split("||")[0])[1]
                    .split(" ")[1]
                    .split(":")[1]
                ),
            ),
            right_entity=EntityAnnotationForRelation(
                text=re.split("(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})", line.split("||")[2])[0]
                .split("=")[1]
                .replace('"', ""),
                start_line=int(
                    re.split("(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})", line.split("||")[2])[1]
                    .split(" ")[0]
                    .split(":")[0]
                ),
                start_word=int(
                    re.split("(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})", line.split("||")[2])[1]
                    .split(" ")[0]
                    .split(":")[1]
                ),
                end_line=int(
                    re.split("(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})", line.split("||")[2])[1]
                    .split(" ")[1]
                    .split(":")[0]
                ),
                end_word=int(
                    re.split("(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})", line.split("||")[2])[1]
                    .split(" ")[1]
                    .split(":")[1]
                ),
            ),
        )
    except (ValueError, IndexError):
        return None


def parse_one_concept(text: str) -> Optional[EntityAnnotation]:
    try:
        return EntityAnnotation(
            label=text.split("||")[1].split("=")[1].replace('"', "").replace("\n", ""),
            text=re.split("(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})", text.split("||")[0])[0]
            .split("=")[1]
            .replace('"', ""),
            start_line=int(
                re.split("(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})", text.split("||")[0])[1]
                .split(" ")[0]
                .split(":")[0]
            ),
            start_word=int(
                re.split("(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})", text.split("||")[0])[1]
                .split(" ")[0]
                .split(":")[1]
            ),
            end_line=int(
                re.split("(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})", text.split("||")[0])[1]
                .split(" ")[1]
                .split(":")[0]
            ),
            end_word=int(
                re.split("(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})", text.split("||")[0])[1]
                .split(" ")[1]
                .split(":")[1]
            ),
        )
    except (ValueError, IndexError):
        return None


def parse_one_assertion(text: str) -> Optional[EntityAnnotation]:
    try:
        return EntityAnnotation(
            label=text.split("||")[2].split("=")[1].replace('"', "").replace("\n", ""),
            text=re.split("(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})", text.split("||")[0])[0]
            .split("=")[1]
            .replace('"', ""),
            start_line=int(
                re.split("(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})", text.split("||")[0])[1]
                .split(" ")[0]
                .split(":")[0]
            ),
            start_word=int(
                re.split("(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})", text.split("||")[0])[1]
                .split(" ")[0]
                .split(":")[1]
            ),
            end_line=int(
                re.split("(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})", text.split("||")[0])[1]
                .split(" ")[1]
                .split(":")[0]
            ),
            end_word=int(
                re.split("(\d{1,6}:\d{1,6} \d{1,6}:\d{1,6})", text.split("||")[0])[1]
                .split(" ")[1]
                .split(":")[1]
            ),
        )
    except (ValueError, IndexError):
        return None


if __name__ == "__main__":
    import os

    text_file = os.path.join("data", "train_data", "partners", "txt", "018636330_DH.txt")
