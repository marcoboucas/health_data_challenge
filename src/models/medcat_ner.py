from collections import defaultdict
import os
import pickle
from typing import List, Optional, Tuple, Union

from medcat.cat import CAT

from src.base.base_ner import BaseNer
from src.types import EntityAnnotation, Token

class MedCATNer(BaseNer):
    def __init__(self, model_path: Optional[str] = 'medmen_wstatus_2021_oct.zip') -> None:
        """Init."""
        super(MedCATNer, self).__init__()
        self.cat = CAT.load_model_pack(model_path)  # type: ignore
        self.label_mapping = defaultdict(lambda: defaultdict(lambda: 0))

    def extract_entities(self, texts: List[str]) -> List[List[EntityAnnotation]]:
        """Extract entities from a list of texts."""
        total_entities = []
        for text in texts:
            current_entities = []
            entities = self.cat.get_entities(text)
            for elem in entities['entities'].values():
                label = self.__find_type_label(elem['types'][0])
                if not label: # Invalid label
                    continue
                name = elem['source_value']
                start_line, start_word = MedCATNer.character_to_line_and_word(text, elem['start'])
                end_line, end_word = MedCATNer.character_to_line_and_word(text, elem['end'])

                current_entities.append(EntityAnnotation(label=label, text=name, start_line=start_line, end_line=end_line, start_word=start_word, end_word=end_word))

            total_entities.append(current_entities)

        return total_entities

    def train(self, annotations: List[List[Union[Token, EntityAnnotation]]]) -> None:
        """Train the model."""
        for annotation in annotations:
                for token in annotation:
                    entities = self.cat.get_entities(token.text)
                    for elem in entities:
                        for type in elem['types']:
                            self.label_mapping[type][token.label] += 1 

    def __find_type_label(self, type: str) -> str:
        """Find type label."""
        best_label = ""
        for label, count in self.label_mapping[type].items():
            if count > self.label_mapping[type].get(best_label, -1):
                best_label = label

        return best_label

    def save_weights(self, weights_path: str) -> None:
        """Save the weights in a file."""
        with open(weights_path, "wb") as file:
            pickle.dump(self.label_mapping, file)

    def load_from_weights(self, weights_path: str) -> None:
        """Load the weights."""
        if not os.path.isfile(weights_path):
            self.logger.warning("No file found here: '%s'", weights_path)
            return
        with open(weights_path, "rb") as file:
            self.label_mapping = pickle.load(file)
        

    @classmethod
    def character_to_line_and_word(cls, text: str, character_position: int) -> Tuple[int, int]:
        """Return the line (starts at 1) and word position (starts at 0) of character_position in text."""
        lines = text.split("\n")

        text_length = 0
        for line_num, line in enumerate(lines):
            if character_position < text_length + (len(line) + 1):
                line_length = 0
                for word_position, word in enumerate(line.split(" ")):
                    if character_position <= text_length + line_length + len(word):
                        return line_num+1, word_position

                    line_length += len(word) + 1 # word length + ' '

            text_length += len(line) + 1 # line length + \n char

        return -1, -1
