from collections import Counter
from dataclasses import dataclass
from itertools import groupby
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from datasets import load_metric
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from src import config
from src.base.base_ner import BaseNer
from src.dataset.dataset_loader import DatasetLoader
from src.types import EntityAnnotation


@dataclass
class BertNerTrainingParameters:
    """Parameters for training"""

    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    weight_decay: float = 0.01


class BertNer(BaseNer):
    """Regex Bert"""

    def __init__(self, device=None):
        """Init"""
        super().__init__()
        # pylint: disable=invalid-name
        self.LABEL_LIST = [
            "O",
            "B_problem",
            "I_problem",
            "B_test",
            "I_test",
            "B_treatment",
            "I_treatment",
        ]
        # pylint: disable=invalid-name
        self.BASE_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
        self.metric = None
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.__load_model()

    def extract_entities(self, texts: List[str]) -> List[List[EntityAnnotation]]:
        """Extract entities from a list of texts."""
        return list(map(self.__extract_entities_one_file, texts))

    def train(
        self,
        dataset_loader_train: DatasetLoader,
        dataset_loader_val: DatasetLoader,
        training_params: BertNerTrainingParameters = BertNerTrainingParameters(),
    ):
        """Train the bert ner model"""
        # First we load the data
        train_data = self.__tokenize_and_align_labels_list(self.__format_data(dataset_loader_train))
        val_data = self.__tokenize_and_align_labels_list(self.__format_data(dataset_loader_val))

        # Then we init the trainer parameters
        args = TrainingArguments(
            "bio_clinical_bert-finetuned-ner",
            evaluation_strategy="epoch",
            learning_rate=training_params.learning_rate,
            per_device_train_batch_size=training_params.batch_size,
            per_device_eval_batch_size=training_params.batch_size,
            num_train_epochs=training_params.num_epochs,
            weight_decay=training_params.weight_decay,
            push_to_hub=False,
        )
        data_collator = DataCollatorForTokenClassification(self.tokenizer)
        self.metric = load_metric("seqeval")

        # We can init the trainer, train the model and save it
        trainer = Trainer(
            self.model,
            args,
            train_dataset=train_data,
            eval_dataset=val_data,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.__compute_metrics,
        )
        trainer.train()
        trainer.save_model(config.NER_BERT_WEIGHTS_FOLDER)

    #############################
    ### For entity extraction ###
    #############################

    def __extract_entities_one_file(self, text: str, batch_size: int = 4) -> List[EntityAnnotation]:
        """Extract entities from one file"""
        # First we must get tokens, predictions and word ids of tokens
        lines = list(map(lambda sentence: sentence.split(" "), text.split("\n")))
        text_entities = []
        for i in range(0, len(lines), batch_size):
            text_entities.extend(self.__extract_entities_one_batch(lines[i : i + batch_size], i))
        return text_entities

    def __extract_entities_one_batch(
        self, lines: List[List[str]], offset: int = 0
    ) -> List[EntityAnnotation]:
        """Exract entities."""
        input_text_tokenized = self.tokenizer(
            lines, truncation=True, padding=True, is_split_into_words=True, return_tensors="pt"
        )
        bert_tokens = list(
            map(
                self.tokenizer.convert_ids_to_tokens,
                input_text_tokenized["input_ids"],
            )
        )
        prediction_model = self.model(input_text_tokenized["input_ids"].to(self.device))
        prediction_logits = prediction_model["logits"].to("cpu")
        predictions = prediction_logits.argmax(dim=2).numpy()
        predictions = np.vectorize(self.LABEL_LIST.__getitem__)(predictions)
        word_ids = [input_text_tokenized.word_ids(i) for i in range(len(bert_tokens))]
        # Now we can format the predictions and extract the entities
        merged_tokens_preds = [
            self.__get_formatted_tokens(token, prediction, word_id)
            for token, prediction, word_id in zip(bert_tokens, predictions, word_ids)
        ]
        all_entities = [
            entity
            for i, (token, pred) in enumerate(merged_tokens_preds)
            for entity in self.__get_entities(token, pred, i + 1)
        ]
        for entity in all_entities:
            entity.start_line += offset
            entity.end_line += offset
        return all_entities

    @staticmethod
    def __get_formatted_tokens(
        tokens: List[str], predictions: List[str], word_ids: List[int]
    ) -> Tuple[List[str], List[str]]:
        """Format text into the format used in the model"""
        final_tokens = []
        final_preds = []
        offset_index = 1
        for i, group in groupby(word_ids[1:]):
            if i is None:
                break
            group = list(group)
            group_tokens = tokens[offset_index : offset_index + len(group)]
            group_preds = predictions[offset_index : offset_index + len(group)]
            if len(group_tokens) == 0:
                break
            token = " ".join(group_tokens).replace(" ##", "")
            label = Counter(group_preds).most_common(1)[0][0]
            offset_index += len(group)
            final_tokens.append(token)
            final_preds.append(label)
        return final_tokens, final_preds

    @staticmethod
    def __get_entities(
        tokens: List[str], predictions: List[str], num_line: int
    ) -> List[EntityAnnotation]:
        """Turn labels into entities"""
        entities = []
        current_pred = None
        current_entity = None
        for i, (token, embed_pred) in enumerate(zip(tokens, predictions)):
            if embed_pred != "O":
                state, pred = embed_pred.split("_")[0], embed_pred.split("_")[1]
                # If a new token must be created
                if current_entity is None or state == "B" or current_pred != pred:
                    if current_entity is not None:
                        entities.append(current_entity)
                    current_entity = EntityAnnotation(
                        label=pred,
                        text=token,
                        start_line=num_line,
                        end_line=num_line,
                        start_word=i,
                        end_word=i,
                    )
                    current_pred = pred
                # If the current token must be updated
                else:
                    current_entity.text += f" {token}"
                    current_entity.end_word = i
            else:
                if current_entity is not None:
                    entities.append(current_entity)
                current_entity = None
                current_pred = None
        if current_entity is not None:
            entities.append(current_entity)
        return entities

    ########################
    ### For the training ###
    ########################

    def __format_data(self, dataset_loader: DatasetLoader) -> Dict[str, Any]:
        """To format data"""
        self.logger.info("Formatting beginning")
        processed_data = []
        nb_tokens_on_multiple_lines = 0
        for elt in dataset_loader:
            words = [s.split(" ") for s in elt.raw_text.split("\n")]
            processed_labels = [["O"] * len(s) for s in words]
            labels = elt.annotation_concept
            for label in labels:
                if label is not None:
                    if label.start_line != label.end_line:
                        nb_tokens_on_multiple_lines += 1
                    begin = True
                    for i in range(label.start_line - 1, label.end_line):
                        for j in range(
                            label.start_word if i == label.start_line - 1 else 0,
                            label.end_word + 1
                            if i == label.end_line - 1
                            else len(processed_labels[i]),
                        ):
                            processed_labels[i][j] = (
                                f"B_{label.label}" if begin else f"I_{label.label}"
                            )
                            begin = False
            processed_data.extend(
                [
                    {"words": sentence, "labels": sentence_labels}
                    for sentence, sentence_labels in zip(words, processed_labels)
                ]
            )
        self.logger.info("Formatting ended")
        return processed_data

    def __tokenize_and_align_labels_list(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """To tokenize and format labels"""
        self.logger.info("Tokenization beginning")
        tokenized_inputs = []
        for row in processed_data:
            tokenized_words = self.tokenizer(
                row["words"], truncation=True, is_split_into_words=True
            )
            word_ids = tokenized_words.word_ids()
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100
                # so they are automatically ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                else:
                    label_ids.append(self.LABEL_LIST.index(row["labels"][word_idx]))
            tokenized_words["labels"] = label_ids
            tokenized_inputs.append(tokenized_words)
        self.logger.info("Tokenization ended")
        return tokenized_inputs

    def __compute_metrics(self, p: Tuple[List[int], List[int]]):
        """To compute metrics for validation"""
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [self.LABEL_LIST[pred] for (pred, lab) in zip(prediction, label) if lab != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.LABEL_LIST[lab] for (_, lab) in zip(prediction, label) if lab != -100]
            for prediction, label in zip(predictions, labels)
        ]

        return self.metric.compute(predictions=true_predictions, references=true_labels)

    #############
    ### Other ###
    #############

    def __load_model(self):
        """To load the model"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.BASE_MODEL)
        try:
            self.model = AutoModelForTokenClassification.from_pretrained(
                config.NER_BERT_WEIGHTS_FOLDER
            ).to(self.device)
            self.logger.info("Loaded the model")
        except OSError:
            self.logger.warning("Didn't manage to load the model, loading the default one")
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.BASE_MODEL, num_labels=len(self.LABEL_LIST)
            )


if __name__ == "__main__":
    import logging
    from pprint import pprint

    logging.basicConfig(level=logging.INFO)
    ner = BertNer()
    logging.info("Training the model !")
    train_set = DatasetLoader(mode="train")
    # val_set = DatasetLoader(mode="val")
    # ner.train(train_set, val_set)
    logging.info("Training done !")
    pprint(ner.extract_entities(["I have pain in my lower body"]))
    # ner.save_weights()
