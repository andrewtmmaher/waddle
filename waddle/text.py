
class Text(object):

    def __init__(self, text):
        self._text = text
        self.unique_characters = list(set(text))
        self.character_to_index = {c: idx for idx, c in enumerate(self.unique_characters)}
        self.index_to_character = {idx: c for idx, c in enumerate(self.unique_characters)}

    def encode_text(self):
        return [self.character_to_index[c] for c in self._text]

    @property
    def vocabulary_size(self):
        return len(self.unique_characters)