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

EVALUATION_CHARACTERS = ['a', 'A', 'R', 'r']


def run(raw_whatsapp_chat, embedding_dimension, window_size,
        number_training_steps, number_evaluation_neighbours):

    whatsapp_chat = clean(raw_whatsapp_chat)

    text = Text(whatsapp_chat)

    model, validation_model = build_embedding_models(text.vocabulary_size, embedding_dimension)

    validation_examples = [text.character_to_index[c] for c in EVALUATION_CHARACTERS]

    similarity_callback = SimilarityCallback(
        number_evaluation_neighbours, validation_examples, text, validation_model)

    context_data = ContextData.from_text(text, window_size)

    train_embedding_model(
        model, similarity_callback, context_data, number_training_steps)


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

    print('Number of training steps: {}'.format(args.number_training_steps))
    print('Embedding dimensionality: {}'.format(args.embedding_dimension))

    print('Chat contains {} different messages'.format(len(chat_data)))
    run(
        chat_data,
        args.embedding_dimension or EMBEDDING_DIMENSION,
        WINDOW_SIZE,
        args.number_training_steps or NUMBER_TRAINING_STEPS,
        NUMBER_EVALUATION_NEIGHBOURS
    )
