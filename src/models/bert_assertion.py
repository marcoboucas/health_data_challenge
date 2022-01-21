"""Assertion BERT model"""
import logging
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from datasets import load_metric
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from src.base.base_assessor import BaseAssessor
from src.dataset.dataset_loader import DatasetLoader
from src.types import EntityAnnotation

LABEL_LIST = ["present", "absent", "possible", "hypothetical", "associated_with_someone_else"]
LABEL_ENCODING_DICT = {
    "present": 0,
    "absent": 1,
    "conditional": 2,
    "possible": 2,
    "hypothetical": 2,
    "associated_with_someone_else": 2,
}
TAG_DUPLICATE = "[duplicate]"
TAG_ENTITY = "[entity]"
MAX_LENGTH = 50


class BertAssessor(BaseAssessor):
    """Bert Assertion NER"""

    def __init__(self) -> None:
        self.model_name = "bvanaken/clinical-assertion-negation-bert"
        self.num_labels = 3

        config = AutoConfig.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name, config=config)
        self.metric = load_metric("seqeval")

        super().__init__()

    def assess_entities(
        self, texts: List[str], entities: List[List[EntityAnnotation]]
    ) -> List[List[EntityAnnotation]]:
        """Assertion prediction functions"""
        assertion_labels = []
        for text, entities_of_text in tqdm(zip(texts, entities), total=len(entities)):
            text = self.format_text(text, entities_of_text).split("\n")
            tokenized_input = self.tokenizer(text, return_tensors="pt", padding=True)
            output = self.model(**tokenized_input)
            predicted_label = np.argmax(output.logits.detach().numpy(), axis=1)
            assertion_labels.append(self.format_assess_entities(predicted_label, entities_of_text))

        return assertion_labels

    def format_assess_entities(
        self, labels: List[int], entities: List[EntityAnnotation]
    ) -> List[EntityAnnotation]:
        """Formats the entities"""
        # we suppose that both list are ordered
        return [
            EntityAnnotation(
                label=self.get_label(label),
                text=entity.text,
                start_line=entity.start_line,
                start_word=entity.start_word,
                end_line=entity.end_line,
                end_word=entity.end_word,
            )
            for label, entity in zip(labels, entities)
        ]

    def train(
        self,
        train_dataset: DatasetLoader,
        val_dataset: DatasetLoader,
        training_params: TrainingArguments,
    ):
        """Train model"""

        train_data = self.tokenize_dataset(train_dataset)
        val_data = self.tokenize_dataset(val_dataset)

        data_collator = DataCollatorForTokenClassification(self.tokenizer, pad_to_multiple_of=8)

        trainer = Trainer(
            self.model,
            args=training_params,
            train_dataset=train_data,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            eval_dataset=val_data,
            compute_metrics=self.compute_metrics,
        )
        trainer.train()

    def tokenize_dataset(self, dataset: DatasetLoader) -> List[dict]:
        """Tokenize dataset"""
        tokenized_dataset = []
        for data in dataset:
            one = self.tokenize_one(
                data.raw_text, data.annotation_concept, data.annotation_assertion
            )
            tokenized_dataset.extend(one)

        return tokenized_dataset

    @staticmethod
    def __get_labels(
        split_sentences, annotation_assertion, indexes
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """Get labels"""
        # list of word labels
        labels = [
            [0 if split_sentences[i][j] is not None else -100 for j in range(len(indexes[i]))]
            for i in range(len(indexes))
        ]

        for entity in annotation_assertion:
            for line in range(entity.start_line - 1, entity.end_line):
                if line < entity.end_line - 1:
                    end_word_idx = len(line)
                else:
                    end_word_idx = entity.end_word
                for word in range(entity.start_word, end_word_idx):
                    if split_sentences[line][word] != TAG_DUPLICATE:
                        labels[line][word] = LABEL_ENCODING_DICT[entity.label]

        sized_labels = []

        for col, _ in enumerate(labels):
            for __ in labels[col]:
                if len(labels[col]) < MAX_LENGTH:
                    sized_labels.append(
                        labels[col] + [0 for ___ in range(MAX_LENGTH - len(labels[col]))]
                    )
                else:
                    sized_labels.append(labels[col][:MAX_LENGTH])
        return labels, sized_labels

    def tokenize_one(
        self,
        text: str,
        assertion_concepts: List[EntityAnnotation],
        annotation_assertion: List[EntityAnnotation],
    ):
        """Tokenize"""
        # concepts "problem" are replaced by "[entity]" in text (as during inference)
        formated_text = self.format_text(text, assertion_concepts, del_duplicates=False)
        # absolute indexes are useful
        indexes = defaultdict(list)
        total = 0
        for i, line in enumerate(formated_text.split("\n")):
            for word in line.split(" "):
                if word != "":
                    indexes[i].append(total)
                    total += 1
        # input text is tokenized
        split_sentences = [
            [word for word in line.split(" ") if word != ""] for line in formated_text.split("\n")
        ]

        tokenized_text = [
            self.tokenizer(
                split_sentence,
                return_tensors="pt",
                max_length=MAX_LENGTH,
                padding="max_length",
                is_split_into_words=True,
                truncation=True,
                add_special_tokens=True,
            )
            for split_sentence in split_sentences
        ]

        sized_labels, labels = self.__get_labels(split_sentences, annotation_assertion, indexes)

        return [
            {
                "input_ids": tokenized_text[i]["input_ids"][0],
                "token_type_ids": tokenized_text[i]["token_type_ids"][0],
                "attention_mask": tokenized_text[i]["attention_mask"][0],
                "labels": sized_labels[i],
            }
            for i in range(len(labels))
        ]

    def compute_metrics(self, input_):
        """Compute the metrics"""
        predictions, labels = input_
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [LABEL_LIST[p] for (p, lab) in zip(prediction, label) if lab != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [LABEL_LIST[lab] for (p, lab) in zip(prediction, label) if lab != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.metric.compute(predictions=true_predictions, references=true_labels)

        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    @staticmethod
    def get_absolute_index(indexes: Dict[int, List[int]], entity) -> Tuple[int, int]:
        """Returns the absolute index"""
        # lines start @ index 1 and word @ index 0
        start = (entity.start_line - 1, entity.start_word)
        end = (entity.end_line - 1, entity.end_word)

        return (indexes[start[0]][start[1]], indexes[end[0]][end[1]])

    @staticmethod
    def get_label(label: int) -> str:
        """Returns label as string
        PRESENT(0), ABSENT(1) or POSSIBLE(2)
        """
        if label == 0:
            return "present"
        if label == 1:
            return "absent"
        return "possible"

    @staticmethod
    def format_text(
        text: str, entities: List[EntityAnnotation], del_duplicates: bool = True
    ) -> str:
        """Returns a text formated for assertion prediction

        Ex: He probably has migraine. (migraine is a problem entity)
        returns: He probably has [entity].
        """
        text_list = [[word for word in line.split(" ") if word != ""] for line in text.split("\n")]
        for entity in entities:
            tag_added = False
            if entity.label == "problem":
                end_word = entity.end_word
                found = ""
                for line_id in range(entity.start_line, entity.end_line + 1):
                    if entity.start_line - entity.end_line == 0:
                        end_index = end_word
                    elif line_id == entity.end_line:
                        end_index = end_word
                    else:
                        end_index = len(text_list[entity.start_line - 1])

                    for word_id in range(entity.start_word, end_index + 1):
                        found += text_list[entity.start_line - 1][word_id] + " "
                        if not tag_added:
                            text_list[entity.start_line - 1][word_id] = TAG_ENTITY
                            tag_added = True
                        elif not del_duplicates:
                            text_list[entity.start_line - 1][word_id] = TAG_DUPLICATE
                        else:
                            text_list[entity.start_line - 1][word_id] = ""

        return "\n".join([" ".join(word) for word in text_list if word != ""])


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    dataset_train = DatasetLoader(mode="train")
    dataset_val = DatasetLoader(mode="val")

    berter_assert = BertAssessor()

    training_arguments = TrainingArguments(
        "assertion-ner-finetuned-ner",
        evaluation_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        push_to_hub=False,
    )
    berter_assert.train(dataset_train, dataset_val, training_arguments)
