"""Main."""
# pylint: disable=too-few-public-methods,no-self-use,too-many-arguments,too-many-locals
import os
from glob import glob
from shutil import rmtree
from typing import Literal, Optional

from tqdm import tqdm

from src import config
from src.dataset.dataset_loader import DatasetLoader
from src.dataset.parser import Parser
from src.evaluation.eval import Evaluator
from src.models import get_assessor, get_ner, get_relation_extractor

DatasetType = Literal["train", "test", "val"]


class CLI:
    """CLI."""

    def run_ner(
        self,
        dataset: DatasetType,
        size: int = -1,
        model_name: str = "regex",
        model_path: Optional[str] = None,
    ):
        """Run the NER model on the dataset."""
        # Prepare the folders and data
        dataset_loader = DatasetLoader(dataset, size)
        ner_results_path = os.path.join(
            config.MODEl_RESULTS_FOLDER,
            dataset,
            "ner_results",
        )
        if os.path.isdir(ner_results_path):
            rmtree(ner_results_path)
        os.makedirs(ner_results_path)

        # Load the model
        ner = get_ner(model_name, model_path)

        # Generate the results
        for dataset_instance in tqdm(
            dataset_loader, desc=f"Prediction with the NER ({model_name})"
        ):
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

    def run_assessor(
        self,
        dataset: DatasetType,
        size: int = -1,
        model_name: str = "random",
        model_path: Optional[str] = None,
    ):
        """Run the NER model on the dataset."""
        # Prepare the folders and data
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

        assert os.path.isdir(
            ner_results_path
        ), "No results for the NER, can't use the concepts then. Please run the ner first"
        if os.path.isdir(assessor_results_path):
            rmtree(assessor_results_path)
        os.makedirs(assessor_results_path)

        # Load the model
        assessor = get_assessor(model_name, model_path)

        parser = Parser()

        # Generate the results
        for i, element_path in enumerate(
            tqdm(
                list(glob(os.path.join(ner_results_path, "*.con"))),
                desc=f"Prediction with the Assessor ({model_name})",
            )
        ):
            if size != -1 and i > size:
                break

            # Element data
            instance_name = os.path.basename(element_path).replace(".con", "")
            instance_text = parser.get_raw_text(
                os.path.join(config.DATA_FOLDERS[dataset], "txt", f"{instance_name}.txt")
            )
            instance_concepts = parser.parse_annotation_concept(element_path)

            # Find the assertions
            assessor_file_path = os.path.join(
                assessor_results_path,
                f"{os.path.basename(element_path).replace('.con', '')}.ast",
            )
            instance_concepts = list(filter(lambda x: x.label == "problem", instance_concepts))
            try:
                assertions = assessor.assess_entities([instance_text], [instance_concepts])[0]
                assessor.assertions_to_file(assertions, assessor_file_path)
            except UnicodeDecodeError:
                logging.warning(
                    "'%s' (%s set) is not readable", instance_name, dataset, exc_info=True
                )

    def run_relation_extractor(
        self,
        dataset: DatasetType,
        size: int = -1,
        model_name: str = "random",
        model_path: Optional[str] = None,
    ):
        """Run the NER model on the dataset."""
        # Prepare the folders and data
        ner_results_path = os.path.join(
            config.MODEl_RESULTS_FOLDER,
            dataset,
            "ner_results",
        )
        relation_results_path = os.path.join(
            config.MODEl_RESULTS_FOLDER,
            dataset,
            "relation_results",
        )

        assert os.path.isdir(
            ner_results_path
        ), "No results for the NER, can't use the concepts then. Please run the ner first"
        if os.path.isdir(relation_results_path):
            rmtree(relation_results_path)
        os.makedirs(relation_results_path)

        # Load the model
        relation_extractor = get_relation_extractor(model_name, model_path)

        parser = Parser()

        # Generate the results
        for i, element_path in enumerate(
            tqdm(
                list(glob(os.path.join(ner_results_path, "*.con"))),
                desc=f"Prediction with the Relation Extractor ({model_name})",
            )
        ):
            if size != -1 and i > size:
                break

            # Element data
            instance_name = os.path.basename(element_path).replace(".con", "")
            instance_text = parser.get_raw_text(
                os.path.join(config.DATA_FOLDERS[dataset], "txt", f"{instance_name}.txt")
            )
            instance_concepts = parser.parse_annotation_concept(element_path)

            # Find the relations
            relation_file_path = os.path.join(
                relation_results_path,
                f"{instance_name}.rel",
            )
            try:
                relations = relation_extractor.find_relations([instance_text], [instance_concepts])[
                    0
                ]
                relation_extractor.relations_to_file(relations, relation_file_path)
            except UnicodeDecodeError:
                logging.warning(
                    "'%s' (%s set) is not readable", instance_name, dataset, exc_info=True
                )

    def run(  # noqa: C901
        self,
        dataset: Literal["train", "test", "val"],
        size: int = -1,
        ner_name: Literal["regex", "medcat", "bert"] = "regex",
        ner_path: Optional[str] = None,
        assessor_name: Literal["random", "bert"] = "random",
        assessor_path: Optional[str] = None,
        relation_extractor_name: Literal["random", "huggingface"] = "random",
        relation_extractor_path: Optional[str] = None,
    ) -> None:
        """Generate all the results.

        `make run`
        """
        self.run_ner(dataset=dataset, size=size, model_name=ner_name, model_path=ner_path)
        self.run_assessor(
            dataset=dataset, size=size, model_name=assessor_name, model_path=assessor_path
        )
        self.run_relation_extractor(
            dataset=dataset,
            size=size,
            model_name=relation_extractor_name,
            model_path=relation_extractor_path,
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
        try:
            data_folder = config.DATA_FOLDERS[dataset]
        except KeyError as exc:
            raise ValueError(f"Wrong value for '{dataset}' (not working with test)") from exc

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
