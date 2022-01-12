"""Dataset loader"""
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List
import pandas as pd
from src import config
from src.dataset.parser import Parser
from src.types import EntityAnnotation, RelationAnnotation


@dataclass
class DataInstance:
    identifier: int
    name: str
    formated_text: Dict[str, str]
    annotation_relation: List[RelationAnnotation]
    annotation_concept: List[EntityAnnotation]
    annotation_assertion: List[EntityAnnotation]


class DatasetLoader:
    """Load the dataset"""

    def __init__(self, train: bool = True) -> None:
        self.columns = ["name", "path", "concept", "ast", "rel"]
        self.parser = Parser()
        if train:
            self.data_frame = pd.read_csv(config.TRAIN_CSV)
        else:
            self.data_frame = pd.read_csv(config.TEST_CSV)

    def __len__(self) -> int:
        """Dataset length"""
        return len(self.data_frame)

    def __getitem__(self, index: int) -> DataInstance:
        """Get an item"""
        patient = self.data_frame.iloc[index]

        return DataInstance(
            identifier=patient.index,
            name=patient["name"],
            formated_text=self.parser.parse_raw_text(patient["path"]),
            annotation_concept=self.parser.parse_annotation_concept(patient["concept"]),
            annotation_relation=self.parser.parse_annotation_relation(patient["rel"]),
            annotation_assertion=self.parser.parse_annotation_concept(patient["ast"]),
        )
