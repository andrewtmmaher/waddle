#!/usr/bin/env python
"""Build a character embedding model."""
import argparse
import os
from model import build_embedding_models, train_embedding_model
from callback import SimilarityCallback
from whatsapp import clean, load_chat_from_path
from text import Text
from data import TrainingData

# Default arguments for the generation of character embeddings.
EMBEDDING_DIMENSION = 2
WINDOW_SIZE = 3
NUMBER_EVALUATION_NEIGHBOURS = 5
NUMBER_TRAINING_STEPS = 10000

EVALUATION_CHARACTERS = ['a', 'A', 'R', 'r']


def run(raw_whatsapp_chat, embedding_dimension, window_size,
        number_training_steps, number_evaluation_neighbours):

    print('Loading and parsing text')
    whatsapp_chat = clean(raw_whatsapp_chat)
    text = Text(whatsapp_chat)

    model, validation_model = build_embedding_models(text.vocabulary_size, embedding_dimension)

    validation_examples = [text.character_to_index[c] for c in EVALUATION_CHARACTERS]

    similarity_callback = SimilarityCallback(
        number_evaluation_neighbours, validation_examples, text, validation_model)

    if os.path.exists('training_data.pkl'):
        print('Loading training data from file')
        training_data = TrainingData.load('training_data.pkl')
    else:
        print('Generating training data from input text')
        training_data = TrainingData.from_text(text, window_size)
        print('Dumping training data to file')
        training_data.dump('training_data.pkl')

    print('Training the embedding')
    train_embedding_model(
        model, training_data, similarity_callback, number_training_steps)


if __name__ == '__main__':

    print('Running character embedding')

    parser = argparse.ArgumentParser()
    parser.add_argument('whatsap_chat_path', type=str)
    parser.add_argument('--embedding_dimension', type=int)
    parser.add_argument('--number_training_steps', type=int)
    parser.add_argument('--test_mode', type=bool)
    args = parser.parse_args()

    print('Loading chat from {}'.format(args.whatsap_chat_path))
    chat_data = load_chat_from_path(args.whatsap_chat_path)

    if args.test_mode:
        print('In test mode.')
        chat_data = chat_data[:100]

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
