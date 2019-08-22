import numpy as np
from tensorflow.keras.preprocessing.sequence import skipgrams


class ContextData(object):

    def __init__(self, target, context, labels):
        """
        Data needed to train the context discrimination model.

        Parameters
        ----------
        target : np.ndarray
            Encoded target character.
        context : np.ndarray
            Encoded character of possible context character.
        labels : np.ndarray
            Positive or negative label indicating whether context character was
            truly found within the context of the target character.
        """
        self.target = target
        self.context = context
        self.labels = labels

    @classmethod
    def from_text(cls, text, window_size):
        couples, labels = skipgrams(
            text.encode_text(), text.vocabulary_size, window_size=window_size)
        target, context = zip(*couples)
        return cls(
            np.array(target, dtype="int32"),
            np.array(context, dtype="int32"),
            labels
        )
