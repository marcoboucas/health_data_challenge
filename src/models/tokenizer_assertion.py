"""Assertion Tokenizer"""
import logging
from itertools import chain
from typing import Dict, List, Optional

from transformers import AutoTokenizer

from src import config
from src.dataset.dataset_loader import DatasetLoader
from src.types import EntityAnnotation


class AssertionTokenizer:
    """Assertion Tokenizer"""

    def __init__(self, base_tokenizer: str = "bvanaken/clinical-assertion-negation-bert") -> None:
        super().__init__()
        self.base_tokenizer = AutoTokenizer.from_pretrained(base_tokenizer)

    def __call__(
        self,
        text: str,
        concepts: List[EntityAnnotation],
        assertions: Optional[List[EntityAnnotation]] = None,
    ) -> List[str]:
        return self.tokenize(text, concepts, assertions)

    def tokenize(
        self,
        text: str,
        concepts: List[EntityAnnotation],
        assertions: Optional[List[EntityAnnotation]] = None,
    ) -> List[str]:
        """Tokenize assertion input (train or inference)"""

        split_text: List[List[str]] = self.__split_into_words(text)
        split_text = self.__tag_concepts(split_text, concepts, del_duplicates=False)
        batch_text = self.__get_batch(split_text=split_text, batch_size=config.BATCH_SIZE)

        tokenized_text = [
            self.base_tokenizer(
                batch,
                return_tensors="pt",
                max_length=config.MAX_LENGTH,
                padding="max_length",
                is_split_into_words=True,
                truncation=True,
                add_special_tokens=True,
            )
            for batch in batch_text
        ]

        if assertions is None:
            return tokenized_text

        labels = self.tokenize_labels(split_text, annotation_assertion=assertions)
        padded_labels = self.__pad_labels(labels)

        return [
            {
                "input_ids": tokenized_text[i]["input_ids"],
                "token_type_ids": tokenized_text[i]["token_type_ids"],
                "attention_mask": tokenized_text[i]["attention_mask"],
                "labels": padded_labels[i],
            }
            for i in range(len(split_text))
        ]

    @staticmethod
    def __get_batch(split_text: List[List[int]], batch_size: int) -> List[List[int]]:
        """Split in batch"""
        total = len(split_text)
        return [split_text[i : min(total, i + batch_size)] for i in range(0, total, batch_size)]

    def tokenize_dataset(self, dataset: DatasetLoader) -> List[Dict[str, List[List[int]]]]:
        """Tokenize dataset"""
        return list(
            chain(
                self.tokenize(data.raw_text, data.annotation_concept, data.annotation_assertion)
                for data in dataset
            )
        )

    def __tag_concepts(
        self,
        split_text: List[List[str]],
        entities: List[EntityAnnotation],
        del_duplicates: bool = False,
    ) -> str:
        """
        Replaces all problem concept entities by a [entity] tag

        replace_concepts('He probably has migraine.' ,..)
        >>>> He probably has [entity].
        """

        for entity in entities:

            tag_added = False
            if entity.label == "problem":
                # we are only interested in problem entities
                found = ""

                for line_id in range(entity.start_line, entity.end_line + 1):
                    # indexes start at 1
                    start_word, end_word = self.__find_start_end(
                        entity, len(split_text[line_id - 1]), line_id
                    )
                    for word_id in range(start_word, end_word):

                        found += split_text[line_id - 1][word_id] + " "

                        if not tag_added:
                            split_text[line_id - 1][word_id] = config.TAG_ENTITY
                            tag_added = True
                        elif not del_duplicates:
                            split_text[line_id - 1][word_id] = config.TAG_DUPLICATE
                        else:
                            split_text[line_id - 1][word_id] = config.TAG_DEL
                if found.strip() != entity.text.strip():
                    logging.warning("Found: '%s' different from initial: '%s'", found, entity.text)
        if del_duplicates:
            return [[token for token in line if token != config.TAG_DEL] for line in split_text]
        return split_text

    def tokenize_labels(self, split_text, annotation_assertion) -> List[List[str]]:
        """Tokenize labels"""

        # deal with special tokens
        labels = [
            [0 if split_text[i][j] is not None else -100 for j in range(len(split_text[i]))]
            for i in range(len(split_text))
        ]

        for entity in annotation_assertion:
            for line_id in range(entity.start_line, entity.end_line + 1):

                start_word, end_word = self.__find_start_end(
                    entity, len(split_text[line_id - 1]), line_id
                )

                for word in range(start_word, end_word):

                    if split_text[line_id - 1][word] != config.TAG_DUPLICATE:
                        labels[line_id - 1][word] = config.LABEL_ENCODING_DICT[entity.label]
        return labels

    @staticmethod
    def __find_start_end(entity: EntityAnnotation, line_length: int, line_id: int):
        """Returns start and end indexes"""

        if entity.start_line == entity.end_line:
            # tagged entities on one line only
            start_word = entity.start_word
            end_word = min(entity.end_word + 1, line_length)

        else:
            # tagged entities on multiple lines
            if line_id == entity.start_line:
                # first line
                start_word = entity.start_word
                end_word = line_length
            elif line_id == entity.end_line:
                # last line
                start_word = 0
                end_word = min(entity.end_word + 1, line_length)
            else:
                # middle
                start_word = 0
                end_word = line_length
        return start_word, end_word

    @staticmethod
    def __pad_labels(labels: List[List[int]]) -> List[List[int]]:
        """Pad labels"""
        padded_labels = []
        for label in labels:
            if len(label) < config.MAX_LENGTH:
                padded_label = label + [0 for _ in range(config.MAX_LENGTH - len(label))]
                padded_labels.append(padded_label)
            else:
                padded_labels.append(label[: config.MAX_LENGTH])
        return padded_labels

    @staticmethod
    def __split_into_words(text: str) -> List[List[str]]:
        """Split text into words"""
        return [[word for word in line.split(" ") if word != ""] for line in text.split("\n")]
