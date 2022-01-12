"""Main."""

import os
from shutil import rmtree
from typing import Literal, Optional

import pandas as pd
from tqdm import tqdm

from src import config
from src.base.base_ner import BaseNer
from src.models.regex_ner import (  # pylint: disable=unused-import
    RegexNer,
    RegexNerWeights,
)


# pylint: disable=too-few-public-methods,no-self-use
class CLI:
    """CLI."""

    def ner(
        self,
        dataset_file: str,
        model_name: Literal["regex"] = "regex",
        model_path: Optional[str] = None,
    ) -> None:
        """Generate the NER results for one dataset.

        `python -m src ner --dataset_file=./data/train.csv --model_path=./weights/ner_regex.pkl`
        """
        # Prepare the folder and data
        df = pd.read_csv(dataset_file)
        ner_results_path = os.path.join(
            config.DATA_FOLDER,
            f"{os.path.basename(dataset_file).replace('.csv', '')}_data",
            "ner_results",
        )
        if os.path.isdir(ner_results_path):
            rmtree(ner_results_path)
        os.makedirs(ner_results_path, exist_ok=True)

        # Load the model
        ner: BaseNer
        if model_name == "regex":

            ner = RegexNer(weights_path=model_path)
        else:
            raise ValueError(f"No '{model_name}' NER model")

        for file_info in tqdm(df.itertuples(), total=df.shape[0]):
            new_file_path = os.path.join(
                ner_results_path,
                os.path.basename(file_info.path).replace(".txt", ".con"),
            )
            try:
                with open(file_info.path, "r", encoding="utf-8") as file:
                    text = file.read()
                entities = ner.extract_entities([text])[0]
                ner.entities_to_file(entities, new_file_path)
            except UnicodeDecodeError:
                logging.warning("'%s' is not readable", file_info.path, exc_info=True)


if __name__ == "__main__":
    import logging

    import fire

    logging.basicConfig(level=logging.INFO)
    fire.Fire(CLI)
