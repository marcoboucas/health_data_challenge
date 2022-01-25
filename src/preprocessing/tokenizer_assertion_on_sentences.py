"""Assertion Tokenizer"""
import logging
from typing import Dict, List, Optional

from transformers import AutoTokenizer

from src import config
from src.dataset.dataset_loader import DatasetLoader
from src.types import EntityAnnotation


class AssertionSentenceTokenizer:
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
        concepts: Optional[List[EntityAnnotation]] = None,
        assertions: Optional[List[EntityAnnotation]] = None,
    ) -> List[str]:
        """Tokenize assertion input (train or inference)"""

        split_text: List[str] = self.__split_into_sentences(text)

        if assertions is not None:
            logging.info("Tokenizer in training mode")

            lines_tagged, lines_labeled = self.__tag_train_samples(split_text, assertions)
            batch_labels = self.__get_batch(lines_labeled, config.BATCH_SIZE)

        else:
            logging.info("Tokenizer in inference mode")
            lines_tagged, lines_concepts = self.__tag_inference_samples(split_text, concepts)

        batch_lines = self.__get_batch(lines_tagged, config.BATCH_SIZE)

        tokenized_text = [
            self.base_tokenizer(
                batch,
                return_tensors="pt",
                max_length=config.MAX_LENGTH,
                padding="max_length",
                is_split_into_words=False,
                truncation=True,
                add_special_tokens=True,
            )
            for batch in batch_lines
        ]

        if assertions is not None:

            return [
                {
                    "input_ids": tokenized_text[i]["input_ids"],
                    "token_type_ids": tokenized_text[i]["token_type_ids"],
                    "attention_mask": tokenized_text[i]["attention_mask"],
                    "labels": batch_labels[i],
                }
                for i, _ in enumerate(batch_labels)
            ]

        return [
            {
                "input_ids": tokenized_text[i]["input_ids"],
                "token_type_ids": tokenized_text[i]["token_type_ids"],
                "attention_mask": tokenized_text[i]["attention_mask"],
            }
            for i, _ in enumerate(batch_lines)
        ], lines_concepts

    def tokenize_train_dataset(self, dataset: DatasetLoader) -> List[Dict[str, List[List[int]]]]:
        """Tokenize dataset"""
        dataset_as_batch = []

        for data in dataset:
            dataset_as_batch.extend(
                self.tokenize(data.raw_text, assertions=data.annotation_assertion)
            )

        return self.__flatten_batchs(dataset_as_batch)

    @staticmethod
    def __get_batch(lines_tagged: List[List[int]], batch_size: int) -> List[List[int]]:
        """Split in batch"""
        total = len(lines_tagged)
        return [lines_tagged[i : min(total, i + batch_size)] for i in range(0, total, batch_size)]

    @staticmethod
    def __flatten_batchs(dataset_as_batch):
        """"""
        flatten_dataset = []
        for batch in dataset_as_batch:
            for line_id in range(config.BATCH_SIZE):
                exists = True
                flatten_item = {
                    "input_ids": None,
                    "token_type_ids": None,
                    "attention_mask": None,
                    "labels": None,
                }
                for key, values in batch.items():
                    if len(values) >= config.BATCH_SIZE:
                        flatten_item[key] = values[line_id]
                    else:
                        exists = False
                if exists:
                    flatten_dataset.append(flatten_item)

        return flatten_dataset

    def __tag_inference_samples(
        self,
        split_text: List[str],
        concepts: List[EntityAnnotation],
    ) -> str:
        """
        Replaces all problem concept entities by a [entity] tag

        replace_concepts('He probably has migraine.' ,..)
        >>>> He probably has [entity].
        """
        lines_tagged = []
        lines_concepts = []

        for concept in concepts:

            tag_added = False
            # we are only interested in problem entities
            found = ""

            for line_id in range(concept.start_line, concept.end_line + 1):
                # indexes start at 1
                start_word, end_word = self.__find_start_end(
                    concept, len(split_text[line_id - 1]), line_id
                )
                split_line = split_text[line_id - 1].split(" ")

                for word_id in range(start_word, end_word):

                    found += split_line[word_id] + " "

                    if not tag_added:
                        split_line[word_id] = config.TAG_ENTITY
                        tag_added = True
                lines_tagged.append(" ".join(split_line))
                lines_concepts.append(concept)

            if (
                found.lower().strip() != concept.text.lower().strip()
                and found.lower().strip() != config.TAG_ENTITY
            ):
                logging.warning("Found: '%s' different from initial: '%s'", found, concept.text)

        return lines_tagged, lines_concepts

    def __tag_train_samples(
        self,
        split_text: List[str],
        assertions: List[EntityAnnotation],
    ) -> str:
        """
        Replaces all problem concept entities by a [entity] tag

        replace_concepts('He probably has migraine.' ,..)
        >>>> He probably has [entity].
        """
        lines_tagged = []
        lines_labeled = []

        for assertion in assertions:

            tag_added = False
            # we are only interested in problem entities
            found = ""

            for line_id in range(assertion.start_line, assertion.end_line + 1):
                # indexes start at 1
                start_word, end_word = self.__find_start_end(
                    assertion, len(split_text[line_id - 1]), line_id
                )
                split_line = split_text[line_id - 1].split(" ")

                for word_id in range(start_word, end_word):

                    found += split_line[word_id] + " "

                    if not tag_added:
                        split_line[word_id] = config.TAG_ENTITY
                        tag_added = True
                lines_tagged.append(" ".join(split_line))
                lines_labeled.append(config.LABEL_ENCODING_DICT[assertion.label])

            if (
                found.lower().strip() != assertion.text.lower().strip()
                and found.lower().strip() != config.TAG_ENTITY
            ):
                logging.warning("Found: '%s' different from initial: '%s'", found, assertion.text)

        return lines_tagged, lines_labeled

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
    def __split_into_sentences(text: str) -> List[str]:
        """Split text into words"""
        return text.split("\n")
