from .model import build_embedding_models, train_embedding_model
from .callback import SimilarityCallback
from .whatsapp import clean
from .text_processing import Text
from .context import ContextData


EMBEDDING_DIMENSION = 64
WINDOW_SIZE = 2
NUMBER_EVALUATION_NEIGHBOURS = 5
NUMBER_TRAINING_STEPS = 10000


def run(raw_whatsapp_chat):

    whatsapp_chat = clean(raw_whatsapp_chat)

    text = Text(whatsapp_chat)

    model, validation_model = build_embedding_models(text.vocabulary_size, EMBEDDING_DIMENSION)

    validation_examples = [text.character_to_index[c] for c in ['a', 'A', 'R', 'r']]

    similarity_callback = SimilarityCallback(
        NUMBER_EVALUATION_NEIGHBOURS, validation_examples, text.index_to_character, validation_model)

    context_data = ContextData.from_text(text, WINDOW_SIZE)

    train_embedding_model(
        model, similarity_callback, context_data, NUMBER_EVALUATION_NEIGHBOURS)
