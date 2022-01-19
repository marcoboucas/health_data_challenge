"""Random Relation Extractor."""

from random import choice
from typing import List

import numpy as np
import torch

from src.base.base_relextractor import BaseRelExtractor
from src.dataset.rel_dataset import RelDataset
from src.types import (
    EntityAnnotation,
    EntityAnnotationForRelation,
    RelationAnnotation,
    RelationValue,
)


class RandomRelExtractor(BaseRelExtractor):
    """Random model for evaluation."""

    def find_relations(
        self, texts: List[str], entities: List[List[EntityAnnotation]]
    ) -> List[List[RelationAnnotation]]:
        """Assess entities.

        :param texts: A list of texts
        :param entities: A list of entities per text
        """
        return [self.__find_relation_one(text, entities) for text, entities in zip(texts, entities)]

    # pylint: disable=unused-argument
    def __find_relation_one(self, text: str, entities: List[EntityAnnotation]):
        """Find one relation type."""
        relations = []

        for _, entities_in_relation in self.find_interesting_lines(text, entities):
            for ent1, ent2 in entities_in_relation:
                relations.append(
                    RelationAnnotation(
                        label=self.__random_relation(),
                        left_entity=EntityAnnotationForRelation(
                            **{k: v for k, v in ent1.__dict__.items() if k != "label"}
                        ),
                        right_entity=EntityAnnotationForRelation(
                            **{k: v for k, v in ent2.__dict__.items() if k != "label"}
                        ),
                    )
                )

        relations = list(filter(lambda x: x != RelationValue.NO_RELATION, relations))
        return relations

    @staticmethod
    def __random_relation() -> str:
        """Return a random relation."""
        return choice(list(RelationValue)).value


if __name__ == "__main__":
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

    training_args = TrainingArguments(
        output_dir=".",
        learning_rate=1e-4,
        weight_decay=0.01,
        num_train_epochs=1,
        seed=666,
    )

    MODEL_NAME = "distilbert-base-uncased"
    PADDING = "max_length"
    SEP_TOKEN = "[SEP]"
    CLS_TOKEN = "[CLS]"
    labels_list = [x.value for x in RelationValue]
    num_labels = len(labels_list)
    id_to_label = dict(enumerate(labels_list))
    label_to_id = {k: i for i, k in enumerate(labels_list)}
    print(labels_list, num_labels)

    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        config=config,
    )

    def compute_metrics(p: EvalPrediction):
        """Compute the metrics."""
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    def preprocess_dataset(dataset):
        """Preprocess a function"""
        outputs = []
        for element in dataset:
            for line, relations in element:
                for relation in relations:

                    final_text = (
                        f"{CLS_TOKEN} {line} {SEP_TOKEN} "
                        f"{relation.left_entity.text} {SEP_TOKEN} {relation.right_entity.text}"
                    )
                    tokenized_text = tokenizer(final_text, padding="max_length", truncation=True)
                    tokenized_text["labels"] = [label_to_id[relation.label.value]]
                    outputs.append(tokenized_text)
        return outputs

    train_dataset = preprocess_dataset(RelDataset("train"))

    set_seed(training_args.seed)
    print(torch.cuda.is_available())
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    train_result = trainer.train()
    metrics = train_result.metrics
    print(metrics)
