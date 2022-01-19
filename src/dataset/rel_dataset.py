"""Relation dataset."""

from collections import defaultdict
from typing import Dict, List, Tuple

from src.dataset.dataset_loader import DatasetLoader
from src.types import RelationAnnotation


class RelDataset(DatasetLoader):
    """Relation dataset."""

    @staticmethod
    def find_interesting_lines_from_relations(
        text: str, relations: List[RelationAnnotation]
    ) -> List[Tuple[str, List[RelationAnnotation]]]:
        """Find the interesting lines for relations."""
        interesting_lines = []
        lines = text.split("\n")

        relations_per_line: Dict[int, List[RelationAnnotation]] = defaultdict(list)
        for relation in relations:
            relations_per_line[relation.left_entity.start_line].append(relation)

        for line_idx, line_relations in relations_per_line.items():
            interesting_lines.append((lines[line_idx - 1], line_relations))
        return interesting_lines

    def __len__(self):
        """Length of the dataset."""
        return self.data_frame.shape[0]

    def __getitem__(self, idx: int):
        """Return all the dataset samples for one text."""
        patient = self.data_frame.iloc[idx]
        text = self.parser.get_raw_text(patient["txt"])
        relations = self.parser.parse_annotation_relation(patient["rel"])

        return self.find_interesting_lines_from_relations(text, relations)
