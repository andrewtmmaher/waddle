#!/usr/bin/env python
import argparse

from model import build_embedding_models, train_embedding_model
from callback import SimilarityCallback
from whatsapp import clean, load_chat_from_path
from text import Text
from context import ContextData


EMBEDDING_DIMENSION = 2
WINDOW_SIZE = 2
NUMBER_EVALUATION_NEIGHBOURS = 5
NUMBER_TRAINING_STEPS = 10000


def run(raw_whatsapp_chat):

    whatsapp_chat = clean(raw_whatsapp_chat)

    text = Text(whatsapp_chat)

    model, validation_model = build_embedding_models(text.vocabulary_size, EMBEDDING_DIMENSION)

    validation_examples = [text.character_to_index[c] for c in ['a', 'A', 'R', 'r']]

    similarity_callback = SimilarityCallback(
        NUMBER_EVALUATION_NEIGHBOURS, validation_examples, text, validation_model)

    context_data = ContextData.from_text(text, WINDOW_SIZE)

    train_embedding_model(
        model, similarity_callback, context_data, NUMBER_EVALUATION_NEIGHBOURS)


if __name__ == '__main__':
    print('Running character embedding')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'whatsap_chat_path', type=str, help='File containing Whatsapp chat log')
    parser.add_argument(
        '--embedding_dimension', type=int, help='Size of embedding')
    parser.add_argument(
        '--number_epochs', type=int, help='Number of samples on which to train')
    args = parser.parse_args()

    print('Loading chat from {}'.format(args.whatsap_chat_path))
    chat_data = load_chat_from_path(args.whatsap_chat_path)

    print('Chat contains {} different messages'.format(len(chat_data)))
    run(chat_data)
