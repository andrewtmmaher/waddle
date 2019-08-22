"""Various functions to clean up a log of Whatsapp messages."""
from collections import Counter


def load_chat_from_path(path):
    with open(path) as f:
        chat = f.readlines()
    return chat


def remove_timestamp(message):
    """Remove the timestamp from a single message."""
    return message[20:]


UNINTERESTING_MESSAGES = [
    '<media omitted>\n',
    'missed voice call\n',
    '\n',
    ''
]


def filter_messages(message):
    return message not in UNINTERESTING_MESSAGES


def _identify_most_common_characters(text, number_characters):
    character_counter = Counter(text).most_common(number_characters)
    return {character for character, __ in character_counter}


def clean(raw_whatsapp_chat):
    raw_whatsapp_chat = [remove_timestamp(message) for message in raw_whatsapp_chat]
    whatsapp_chat = [message for message in raw_whatsapp_chat if filter_messages(message[14:])]

    text = ''.join([character for message in whatsapp_chat for character in message])
    most_common_characters = _identify_most_common_characters(text, 200)

    return [character for character in text if character in most_common_characters]
