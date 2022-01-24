"""Assertion BERT model"""
import logging
from typing import List

import numpy as np
from datasets import load_metric
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from src import config
from src.base.base_assessor import BaseAssessor
from src.dataset.dataset_loader import DatasetLoader
from src.models.tokenizer_assertion import AssertionTokenizer
from src.types import EntityAnnotation


class BertAssessor(BaseAssessor):
    """Bert Assertion NER"""

    def __init__(self) -> None:
        self.model_name = "bvanaken/clinical-assertion-negation-bert"
        self.num_labels = 3

        self.config = AutoConfig.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
        )
        self.tokenizer = AssertionTokenizer(base_tokenizer=self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name, config=self.config
        )
        self.metric = load_metric("seqeval")

        super().__init__()

    def load_weights(self, path: str):
        """Loads the weights"""
        self.model.load_state_dict(path)

    def assess_entities(
        self, texts: List[str], concepts: List[List[EntityAnnotation]]
    ) -> List[List[EntityAnnotation]]:
        """Assertion prediction functions"""
        pred_assertions = []

        for text, entities in tqdm(zip(texts, concepts), total=len(concepts)):
            classified_tokens = []
            tokenized_text = self.tokenizer(text=text, concepts=entities)
            for batch in tqdm(tokenized_text):
                output = self.model(**batch)
                output_as_np = output.logits.detach().numpy()
                for i in range(output_as_np.shape[0]):
                    line_output = output_as_np[i, :, :]
                    classified_tokens.append(np.argmax(line_output, axis=1))

            pred_assertions.append(self.__format_assertions(classified_tokens, entities))

        return pred_assertions

    def train(
        self,
        train_dataset: DatasetLoader,
        training_params: TrainingArguments,
    ):
        """Train model"""

        train_data = self.tokenizer.tokenize_dataset(train_dataset)

        # val_data = self.tokenize_dataset(val_dataset)

        data_collator = DataCollatorForTokenClassification(
            self.tokenizer.base_tokenizer, pad_to_multiple_of=config.BATCH_SIZE, padding=False
        )

        trainer = Trainer(
            self.model,
            args=training_params,
            train_dataset=train_data,
            tokenizer=self.tokenizer.base_tokenizer,
            data_collator=data_collator,
        )
        # eval_dataset=val_data,
        # compute_metrics=self.compute_metrics,
        trainer.train()

    @staticmethod
    def __format_assertions(classified_tokens: List[List[int]], entities: List[EntityAnnotation]):
        """Format assertions"""
        return [
            EntityAnnotation(
                label=config.LABEL_LIST[classified_tokens[entity.start_line][entity.start_word]],
                text=entity.text,
                start_line=entity.start_line,
                start_word=entity.start_word,
                end_line=entity.end_line,
                end_word=entity.end_word,
            )
            for entity in entities
            if entity.label == "problem" and entity.start_word < config.MAX_LENGTH
        ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    dataset_train = DatasetLoader(mode="train")
    dataset_val = DatasetLoader(mode="val")

    berter_assert = BertAssessor()

    training_arguments = TrainingArguments(
        "assertion-ner-finetuned-ner",
        learning_rate=1e-4,
        per_device_train_batch_size=config.BATCH_SIZE,
        num_train_epochs=10,
        weight_decay=0.01,
        push_to_hub=False,
    )

    berter_assert.train(dataset_train, training_arguments)
