"""Random Relation Extractor."""

import os
from collections import defaultdict
from pprint import pprint
from random import choice, shuffle
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.base.base_relation_extractor import BaseRelExtractor
from src.config import DEFAULT_RELATION_WEIGHTS_FOLDER
from src.dataset.rel_dataset import RelDataset
from src.types import (
    EntityAnnotation,
    EntityAnnotationForRelation,
    RelationAnnotation,
    RelationValue,
)


# pylint: disable=attribute-defined-outside-init,too-many-instance-attributes
class BertRelExtractor(BaseRelExtractor):
    """Bert model for evaluation."""

    padding: str = "max_length"
    sep_token: str = "[SEP]"
    cls_token: str = "[CLS]"

    def __init__(self, weights_path: Optional[str] = None, device=None) -> None:
        """Init."""
        super().__init__()

        # About the labels
        self.labels_list = [x.value for x in RelationValue]
        self.num_labels = len(self.labels_list)
        self.id_to_label = dict(enumerate(self.labels_list))
        self.label_to_id = {k: i for i, k in enumerate(self.labels_list)}
        self.correct_outputs = {
            "problem": [
                self.label_to_id[x.value] for x in [RelationValue.PIP, RelationValue.NO_RELATION]
            ],
            "treatment": [
                self.label_to_id[x.value]
                for x in [
                    RelationValue.NO_RELATION,
                    RelationValue.TrAP,
                    RelationValue.TrCP,
                    RelationValue.TrIP,
                    RelationValue.TrNAP,
                    RelationValue.TrWP,
                ]
            ],
            "test": [
                self.label_to_id[x.value]
                for x in [RelationValue.NO_RELATION, RelationValue.TeCP, RelationValue.TeRP]
            ],
        }

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.logger.info("Running the model on the device: '%s'", self.device)

        # Load the model if required
        if weights_path is not None and os.path.isdir(weights_path):
            self.load_model(weights_path)
        else:
            self.logger.warning("The model has not be loaded, you should train it")

    def load_model(self, model_path: str) -> None:
        """Load the model."""
        self.logger.info("Loading the model from pretrained weights...")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def find_relations(
        self, texts: List[str], entities: List[List[EntityAnnotation]]
    ) -> List[List[RelationAnnotation]]:
        """Assess entities.

        :param texts: A list of texts
        :param entities: A list of entities per text
        """

        return [self.__find_relation_one(text, entities) for text, entities in zip(texts, entities)]

    # pylint: disable=too-many-locals
    def __find_relation_one(
        self,
        text: str,
        entities: List[EntityAnnotation],
        batch_size: int = 16,
    ) -> List[RelationAnnotation]:
        """Find all annotations for one text."""
        # Get the inputs
        interesting_lines = self.find_interesting_lines_from_entities(text, entities)
        self.logger.info(
            "Found %i interesting lines, for %i entities",
            len(interesting_lines),
            sum([len(x[1]) for x in interesting_lines]),
        )
        text_inputs: List[str] = []
        entities_pairs = []
        for line, line_entities in interesting_lines:
            for ent1, ent2 in line_entities:
                text_inputs.append(self.__get_one_text(line, ent1, ent2))
                entities_pairs.append((ent1, ent2))

        # Call the model
        self.model.eval()
        self.logger.info("We will run the model on %i texts", len(text_inputs))
        with torch.no_grad():
            outputs = []

            for i in range(0, len(text_inputs), batch_size):
                model_inputs = self.tokenizer(
                    text_inputs[i : i + batch_size],
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )
                model_outputs = self.model(model_inputs["input_ids"].to(self.device))

                # Add the correct output in the list
                # Because we know if it a problem-problem, problem-test, problem-treatment
                for j, model_output in enumerate(model_outputs.logits.detach().cpu()):
                    entity_pair = entities_pairs[i + j]

                    types = {x.label for x in entity_pair}

                    if len(types) == 1 and "problem" in types:
                        possible_values = self.correct_outputs["problem"]
                    elif len(types) == 2 and "treatment" in types:
                        possible_values = self.correct_outputs["treatment"]
                    elif len(types) == 2 and "test" in types:
                        possible_values = self.correct_outputs["test"]
                    else:
                        raise ValueError("Fuck...")
                    # possible_values = list(range(len(self.id_to_label)))
                    corresponding_label = possible_values[
                        model_output[possible_values].argmax().item()
                    ]
                    outputs.append(self.id_to_label[corresponding_label])
                    # Get the possible values:
        assert len(outputs) == len(entities_pairs)
        results = []
        for output, (ent1, ent2) in zip(outputs, entities_pairs):
            results.append(
                RelationAnnotation(
                    label=output,
                    left_entity=EntityAnnotationForRelation(
                        **{k: v for k, v in ent1.__dict__.items() if k != "label"}
                    ),
                    right_entity=EntityAnnotationForRelation(
                        **{k: v for k, v in ent2.__dict__.items() if k != "label"}
                    ),
                )
            )
        return results

    def __get_one_text(
        self,
        line: str,
        left_entity: Union[EntityAnnotationForRelation, EntityAnnotation],
        right_entity: Union[EntityAnnotationForRelation, EntityAnnotation],
    ) -> str:
        """Get the text for one relation."""
        return (
            f"{self.cls_token} {line} {self.sep_token} "
            f"{left_entity.text} {self.sep_token} {right_entity.text}"
        )

    def __get_one_input(
        self,
        line: str,
        left_entity: Union[EntityAnnotationForRelation, EntityAnnotation],
        right_entity: Union[EntityAnnotationForRelation, EntityAnnotation],
        label: Optional[str] = None,
    ) -> Dict:
        """Get one input from the needed data."""
        final_text = self.__get_one_text(line, left_entity, right_entity)
        tokenized_text = self.tokenizer(
            final_text,
            padding="max_length",
            truncation=True,
            return_tensors=None if label else "pt",
        )
        if label:
            tokenized_text["labels"] = [self.label_to_id[label]]
        return tokenized_text

    def preprocess_dataset(self, dataset, size: int = -1) -> List[Dict[str, Any]]:
        """Preprocess a function"""
        outputs = []
        for interesting_lines, concepts, lines in dataset:

            # Add the real relations
            real_relations: Dict[str, Any] = {}
            for line, relations in interesting_lines:
                for relation in relations:
                    real_relations[
                        self.__get_one_text(line, relation.left_entity, relation.right_entity)
                    ] = self.__get_one_input(
                        line,
                        relation.left_entity,
                        relation.right_entity,
                        label=relation.label.value,
                    )

            # Find plausible false relations
            concepts_per_line: Dict[str, List[EntityAnnotation]] = defaultdict(list)
            for concept in concepts:
                concepts_per_line[concept.start_line - 1].append(concept)
            concepts_per_line = dict(concepts_per_line)
            concepts_per_line = {k: v for k, v in concepts_per_line.items() if len(v) > 2}

            # Add false relations, to train the model how to react to false relations
            false_relations: Dict[str, Any] = {}
            for _ in range(len(real_relations) * 3):
                if len(false_relations) == len(real_relations) or len(concepts_per_line) == 0:
                    break

                line_idx = choice(list(concepts_per_line.keys()))
                line = lines[line_idx]

                concept1 = choice(concepts_per_line[line_idx])
                concept2 = choice(concepts_per_line[line_idx])
                if concept1.start_word == concept2.start_word:
                    continue
                key = self.__get_one_text(line, concept1, concept2)
                if key not in real_relations:
                    false_relations[key] = self.__get_one_input(
                        line,
                        concept1,
                        concept1,
                        label=RelationValue.NO_RELATION.value,
                    )
            relations = list({**real_relations, **false_relations}.values())
            shuffle(relations)
            outputs.extend(relations)
            if size != -1 and len(outputs) > size:
                break
        return outputs

    # pylint: disable=too-many-locals,too-many-arguments
    def train(
        self,
        model_name: str = "distilbert-base-uncased",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        epochs: int = 5,
        batch_size: int = 128,
        evaluate: bool = False,
        plot_fig: bool = False,
    ) -> None:
        """Train the model."""
        self.logger.info("Training the Huggingface Model for Relations")

        # Setup the config, model and tokenizer
        training_args = TrainingArguments(
            output_dir=DEFAULT_RELATION_WEIGHTS_FOLDER,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_train_epochs=epochs,
            seed=666,
            per_device_train_batch_size=batch_size,
        )
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=self.num_labels,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config,
        )
        data_collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8)

        # Load the training set
        train_dataset = self.preprocess_dataset(RelDataset("train"))
        self.logger.info("Training on %i elements", len(train_dataset))
        val_dataset = self.preprocess_dataset(RelDataset("val"))
        self.logger.info("Evaluation on %i elements", len(val_dataset))
        # Training !!
        set_seed(training_args.seed)
        self.logger.info("Cuda device availability: %s", torch.cuda.is_available())

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            # tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        train_result = trainer.train()
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        if evaluate:
            val_dataset = self.preprocess_dataset(RelDataset("val"))

            true_values = np.array([self.id_to_label[x["labels"][0]] for x in val_dataset])
            predictions = trainer.predict(val_dataset, metric_key_prefix="predict").predictions
            predictions = np.argmax(predictions, axis=1)
            predictions = [self.id_to_label[x] for x in predictions]

            print(classification_report(true_values, predictions, labels=self.labels_list))

            cm = confusion_matrix(
                true_values, predictions, labels=self.labels_list, normalize="true"
            )
            print(cm)
            if plot_fig:
                ConfusionMatrixDisplay(
                    confusion_matrix=cm,
                    display_labels=self.labels_list,
                ).plot()

                plt.show()

    @staticmethod
    def compute_metrics(p: EvalPrediction):
        """Compute the metrics."""
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    TRAIN_MODEL = True
    if TRAIN_MODEL:

        relation_extractor = BertRelExtractor()

        relation_extractor.train(
            model_name="distilbert-base-uncased",
            learning_rate=1e-4,
            weight_decay=0.01,
            epochs=5,
            batch_size=8,
            evaluate=True,
            plot_fig=True,
        )
    else:

        relation_extractor = BertRelExtractor(weights_path="./weights/relextractor_bert")
        pprint(
            relation_extractor.find_relations(
                texts=["hypertension was controlled on hydrochlorothiazide"],
                entities=[
                    [
                        EntityAnnotation("problem", "hypertension", 1, 1, 0, 0),
                        EntityAnnotation("problem", "hydrochlorothiazide", 1, 1, 4, 4),
                    ]
                ],
            )
        )
