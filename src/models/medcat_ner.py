from typing import Optional, Set, Union, List, Tuple# Int, Tuple

from medcat.cat import CAT

from src.base.base_ner import BaseNer
from src.types import EntityAnnotation, Token

class MedCATNer(BaseNer):
    def __init__(self, model_path: Optional[str] = 'abc') -> None:
        """Init."""

        super(MedCATNer, self).__init__()
        self.cat = CAT.load_model_pack(model_path)

    def extract_entities(self, texts: List[str]) -> List[List[EntityAnnotation]]:
        total_entities = []
        for text in texts:
            current_entities = []
            entities = self.cat.get_entities(text)
            for elem in entities['entities'].values():
                label = elem['types'][0]
                name = elem['source_value']
                start_line, start_word = MedCATNer.character_to_line_and_word(text, elem['start'])
                end_line, end_word = MedCATNer.character_to_line_and_word(text, elem['end'])

                current_entities.append(EntityAnnotation(label=label, text=name, start_line=start_line, end_line=end_line, start_word=start_word, end_word=end_word))

            total_entities.append(current_entities)

        return total_entities

    @classmethod
    def character_to_line_and_word(cls, text: str, character_position: int) -> Tuple[int, int]:
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


    #def __filter_entities(self):
