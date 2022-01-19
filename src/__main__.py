"""Main."""
# pylint: disable=too-few-public-methods,no-self-use,too-many-arguments,too-many-locals
import os
from shutil import rmtree
from typing import Literal, Optional

from tqdm import tqdm

from src import config
from src.dataset.dataset_loader import DataInstance, DatasetLoader
from src.evaluation.eval import Evaluator
from src.models import get_assessor, get_ner, get_relation_extractor


class CLI:
    """CLI."""

    def run(
        self,
        dataset: Literal["train", "test", "val"],
        size: int = -1,
        ner_name: Literal["regex", "medcat", "bert"] = "regex",
        ner_path: Optional[str] = None,
        assessor_name: Literal["random", "bert"] = "random",
        relation_extractor_name: Literal["random", "huggingface"] = "random",
        relation_extractor_path: Optional[str] = None,
    ) -> None:
        """Generate the NER results for one dataset.

        `make run`
        """
        # Prepare the folders and data
        dataset_loader = DatasetLoader(dataset, size)
        ner_results_path = os.path.join(
            config.MODEl_RESULTS_FOLDER,
            dataset,
            "ner_results",
        )
        assessor_results_path = os.path.join(
            config.MODEl_RESULTS_FOLDER,
            dataset,
            "assessor_results",
        )
        relation_results_path = os.path.join(
            config.MODEl_RESULTS_FOLDER,
            dataset,
            "relation_results",
        )
        if os.path.isdir(ner_results_path):
            rmtree(ner_results_path)
        os.makedirs(ner_results_path)
        if os.path.isdir(assessor_results_path):
            rmtree(assessor_results_path)
        os.makedirs(assessor_results_path)
        if os.path.isdir(relation_results_path):
            rmtree(relation_results_path)
        os.makedirs(relation_results_path)

        # Load the models (NER, assessor, relation extractor)
        ner = get_ner(ner_name, ner_path)
        assessor = get_assessor(assessor_name)
        relextractor = get_relation_extractor(relation_extractor_name, relation_extractor_path)

        dataset_instance: DataInstance
        for dataset_instance in tqdm(dataset_loader):
            # Find the concepts
            ner_file_path = os.path.join(
                ner_results_path,
                f"{dataset_instance.name.replace('.txt', '')}.con",
            )
            try:
                concepts = ner.extract_entities([dataset_instance.raw_text])[0]
                ner.entities_to_file(concepts, ner_file_path)
            except UnicodeDecodeError:
                logging.warning(
                    "'%s' (%s set) is not readable", dataset_instance.name, dataset, exc_info=True
                )

            # Find the assertions
            assessor_file_path = os.path.join(
                assessor_results_path,
                f"{dataset_instance.name.replace('.txt', '')}.ast",
            )
            concepts = list(filter(lambda x: x.label == "problem", concepts))
            try:
                assertions = assessor.assess_entities([dataset_instance.raw_text], [concepts])[0]
                assessor.assertions_to_file(assertions, assessor_file_path)
            except UnicodeDecodeError:
                logging.warning(
                    "'%s' (%s set) is not readable", dataset_instance.name, dataset, exc_info=True
                )

            # Find the relations
            relation_file_path = os.path.join(
                relation_results_path,
                f"{dataset_instance.name.replace('.txt', '')}.rel",
            )
            try:
                relations = relextractor.find_relations([dataset_instance.raw_text], [concepts])[0]
                relextractor.relations_to_file(relations, relation_file_path)
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
            assertion_prediction_dir=os.path.join(results_path, "assessor_results"),
            relation_annotation_dir=os.path.join(data_folder, "rel"),
            relation_prediction_dir=os.path.join(results_path, "relation_results"),
            entries_dir=os.path.join(data_folder, "txt"),
        )

        evaluator.evaluate()


if __name__ == "__main__":
    import logging

    import fire

    logging.basicConfig(level=logging.INFO)
    fire.Fire(CLI)
