"""Main."""

import os
from shutil import rmtree
from typing import Literal, Optional

from tqdm import tqdm

from src import config
from src.base.base_ner import BaseNer
from src.dataset.dataset_loader import DataInstance, DatasetLoader
from src.evaluation.eval import Evaluator
from src.models.medcat_ner import MedCATNer
from src.models.regex_ner import RegexNer


# pylint: disable=too-few-public-methods,no-self-use
class CLI:
    """CLI."""

    def ner(
        self,
        dataset: Literal["train", "test", "val"],
        model_name: Literal["regex", "medcat"] = "regex",
        model_path: Optional[str] = None,
    ) -> None:
        """Generate the NER results for one dataset.

        `python -m src ner --dataset=train --model_name=regex --model_path=./weights/ner_regex.pkl`
        """
        # Prepare the folder and data
        dataset_loader = DatasetLoader(dataset)
        ner_results_path = os.path.join(
            config.MODEl_RESULTS_FOLDER,
            dataset,
            "ner_results",
        )
        if os.path.isdir(ner_results_path):
            rmtree(ner_results_path)
        os.makedirs(ner_results_path, exist_ok=True)

        # Load the model
        ner: BaseNer
        if model_name == "regex":

            ner = RegexNer(weights_path=model_path or config.NER_REGEX_WEIGHTS_FILE)
        elif model_name == "medcat":

            ner = MedCATNer(weights_path=model_path or config.NER_MEDCAT_WEIGHTS_FILE)
        else:
            raise ValueError(f"No '{model_name}' NER model")

        dataset_instance: DataInstance
        for dataset_instance in tqdm(dataset_loader):
            new_file_path = os.path.join(
                ner_results_path,
                f"{dataset_instance.name}.con",
            )
            try:
                entities = ner.extract_entities([dataset_instance.raw_text])[0]
                ner.entities_to_file(entities, new_file_path)
            except UnicodeDecodeError:
                logging.warning(
                    "'%s' (%s set) is not readable", dataset_instance.name, dataset, exc_info=True
                )

    def eval(self, dataset: str = "train", results_path: Optional[str] = None) -> None:
        """Evaluate the model results.

        `python -m src eval --dataset=train`
        """
        if results_path is None:
            results_path = os.path.join(
                config.MODEl_RESULTS_FOLDER,
                dataset,
            )
        evaluator: Evaluator
        if dataset == "train":
            data_folder = config.TRAIN_DATA_FOLDER
        elif dataset == "val":
            data_folder = config.VAL_DATA_FOLDER
        else:
            raise ValueError(f"Wrong value for '{dataset}' (not working with test)")

        evaluator = Evaluator(
            concept_annotation_dir=os.path.join(data_folder, "concept"),
            concept_prediction_dir=os.path.join(results_path, "ner_results"),
            assertion_annotation_dir=os.path.join(data_folder, "ast"),
            assertion_prediction_dir="",
            relation_annotation_dir=os.path.join(data_folder, "rel"),
            relation_prediction_dir="",
            entries_dir=os.path.join(data_folder, "txt"),
        )

        evaluator.evaluate()


if __name__ == "__main__":
    import logging

    import fire

    logging.basicConfig(level=logging.INFO)
    fire.Fire(CLI)
