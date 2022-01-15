"""Dataset loader"""
from dataclasses import asdict, dataclass
from typing import Dict, List

import pandas as pd

from src import config
from src.dataset.parser import Parser
from src.types import EntityAnnotation, RelationAnnotation


@dataclass
class DataInstance:
    """Dataset Instance."""

    identifier: int
    name: str
    raw_text: str
    formated_text: Dict[str, str]
    annotation_relation: List[RelationAnnotation]
    annotation_concept: List[EntityAnnotation]
    annotation_assertion: List[EntityAnnotation]


class DatasetLoader:
    """Load the dataset"""

    def __init__(self, train: bool = True) -> None:
        self.columns = ["name", "path", "concept", "ast", "rel"]
        self.parser = Parser()
        self.data_frame = pd.read_csv(config.TRAIN_CSV if train else config.TEST_CSV, index_col=0)

    def __len__(self) -> int:
        """Dataset length"""
        return len(self.data_frame)

    def __getitem__(self, index: int) -> DataInstance:
        """Get an item"""
        patient = self.data_frame.iloc[index]
        return DataInstance(
            identifier=str(patient.name),
            name=patient["name"],
            formated_text=self.parser.parse_raw_text(patient["path"]),
            raw_text=self.parser.get_raw_text(patient["path"]),
            annotation_concept=self.parser.parse_annotation_concept(patient["concept"]),
            annotation_relation=self.parser.parse_annotation_relation(patient["rel"]),
            annotation_assertion=self.parser.parse_annotation_assertion(patient["ast"]),
        )


if __name__ == "__main__":
    import json

    dataset = DatasetLoader()
    print(f"Lenght of the dataset: {len(dataset)}")

    with open("./test.json", "w", encoding="utf-8") as file:
        ele = dataset[0]
        json.dump(asdict(ele), file, indent=2, default=str)
    print("Saved in ./test.json")
