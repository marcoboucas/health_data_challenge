"""Dataset loader"""
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Literal

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
    annotation_relation: List[RelationAnnotation] = field(default_factory=list)
    annotation_concept: List[EntityAnnotation] = field(default_factory=list)
    annotation_assertion: List[EntityAnnotation] = field(default_factory=list)


class DatasetLoader:
    """Load the dataset"""

    def __init__(self, mode: Literal["train", "val", "test"] = "train", size: int = -1) -> None:
        self.size = size
        self.columns = ["name", "path", "concept", "ast", "rel"]
        self.parser = Parser()
        if mode == "train":
            self.data_frame = pd.read_csv(config.TRAIN_CSV).head(self.size)
        elif mode == "val":
            self.data_frame = pd.read_csv(config.VAL_CSV).head(self.size)
        else:
            self.data_frame = pd.read_csv(config.TEST_CSV).head(self.size)

    def __len__(self) -> int:
        """Dataset length"""
        return len(self.data_frame)

    def __getitem__(self, index: int) -> DataInstance:
        """Get an item"""
        patient = self.data_frame.iloc[index]
        return DataInstance(
            identifier=str(patient.name),
            name=patient["name"],
            formated_text=self.parser.parse_raw_text(patient["txt"]),
            raw_text=self.parser.get_raw_text(patient["txt"]),
            annotation_concept=self.parser.parse_annotation_concept(patient["concept"])
            if "concept" in patient
            else [],
            annotation_relation=self.parser.parse_annotation_relation(patient["rel"])
            if "rel" in patient
            else [],
            annotation_assertion=self.parser.parse_annotation_assertion(patient["ast"])
            if "ast" in patient
            else [],
        )


if __name__ == "__main__":
    import json

    dataset = DatasetLoader()
    print(f"Lenght of the dataset: {len(dataset)}")

    with open("./test.json", "w", encoding="utf-8") as file:
        ele = dataset[0]
        json.dump(asdict(ele), file, indent=2, default=str)
    print("Saved in ./test.json")
