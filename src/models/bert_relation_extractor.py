"""Random Relation Extractor."""

import os
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm
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


# pylint: disable=attribute-defined-outside-init
class BertRelExtractor(BaseRelExtractor):
    """Bert model for evaluation."""

    padding: str = ("max_length",)
    sep_token: str = ("[SEP]",)
    cls_token: str = "[CLS]"

    def __init__(self, weights_path: Optional[str] = None, device=None) -> None:
        """Init."""
        super().__init__()

        # About the labels
        self.labels_list = [x.value for x in RelationValue]
        self.num_labels = len(self.labels_list)
        self.id_to_label = dict(enumerate(self.labels_list))
        self.label_to_id = {k: i for i, k in enumerate(self.labels_list)}

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

    def __find_relation_one(
        self, text: str, entities: List[EntityAnnotation], verbose: bool = False
    ) -> List[RelationAnnotation]:
        """Find all annotations for one text."""
        # Get the inputs
        interesting_lines = self.find_interesting_lines_from_entities(text, entities)
        inputs = []
        entities_pairs = []
        for line, line_entities in interesting_lines:
            for ent1, ent2 in line_entities:
                inputs.append(self.__get_one_input(line, ent1, ent2))
                entities_pairs.append((ent1, ent2))

        # Call the model
        self.model.eval()
        self.logger.debug("We will run the model on %i texts", len(inputs))
        with torch.no_grad():
            outputs = []
            # TODO: Use a dataloader to speed up the process (batch)
            for input_ in tqdm(inputs, disable=not verbose):
                output = self.model(**input_.to(self.device))
                outputs.append(self.id_to_label[output.logits.detach().cpu().argmax().item()])

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

    def __get_one_input(
        self,
        line: str,
        left_entity: Union[EntityAnnotationForRelation, EntityAnnotation],
        right_entity: Union[EntityAnnotationForRelation, EntityAnnotation],
        label: Optional[str] = None,
    ) -> Dict:
        """Get one input from the needed data."""
        final_text = (
            f"{self.cls_token} {line} {self.sep_token} "
            f"{left_entity.text} {self.sep_token} {right_entity.text}"
        )
        tokenized_text = self.tokenizer(
            final_text, padding="max_length", truncation=True, return_tensors="pt"
        )
        if label:
            tokenized_text["labels"] = [self.label_to_id[label]]
        return tokenized_text

    def preprocess_dataset(self, dataset, size: int = -1):
        """Preprocess a function"""
        outputs = []
        for element in dataset:
            for line, relations in element:
                for relation in relations:
                    outputs.append(
                        self.__get_one_input(
                            line,
                            relation.left_entity,
                            relation.right_entity,
                            label=relation.label.value,
                        )
                    )
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
        batch_size: int = 8,
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

        # Training !!
        set_seed(training_args.seed)
        self.logger.info("Cuda device availability: %s", torch.cuda.is_available())
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=None,
            compute_metrics=self.compute_metrics,
            tokenizer=self.tokenizer,
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
            labels_without_no_relation = list(
                filter(lambda x: x != "NO_RELATION", self.labels_list)
            )
            val_dataset = self.preprocess_dataset(RelDataset("val"))

            true_values = np.array([self.id_to_label[x["labels"][0]] for x in val_dataset])
            predictions = trainer.predict(val_dataset, metric_key_prefix="predict").predictions
            predictions = np.argmax(predictions, axis=1)
            predictions = [self.id_to_label[x] for x in predictions]

            print(
                classification_report(true_values, predictions, labels=labels_without_no_relation)
            )

            cm = confusion_matrix(
                true_values, predictions, labels=labels_without_no_relation, normalize="true"
            )
            print(cm)
            if plot_fig:
                ConfusionMatrixDisplay(
                    confusion_matrix=cm,
                    display_labels=labels_without_no_relation,
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

    logging.basicConfig(level=logging.DEBUG)
    relation_extractor = BertRelExtractor(weights_path="./weights/relextractor_bert")

    relation_extractor.train(
        model_name="distilbert-base-uncased",
        learning_rate=1e-4,
        weight_decay=0.01,
        epochs=5,
        batch_size=8,
        evaluate=True,
        plot_fig=True,
    )
    """
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
    """
