import os
import pathlib
import re
import warnings
from typing import List

import sentence_transformers

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
MESSAGE_PAD_TOKENS = int(os.getenv("MESSAGE_PAD_TOKENS", 3))
TOKENS_TO_TRIGGER_SUMMARY = int(os.getenv("TOKENS_TO_TRIGGER_SUMMARY", 150))
WORD_TO_TOKEN_RATIO = 1.5


def get_messages_size(messages: List[dict]) -> int:
    num_tokens = 0
    for message in messages:
        num_tokens += MESSAGE_PAD_TOKENS
        for value in message.values():
            num_tokens += int(len(value.split()) * WORD_TO_TOKEN_RATIO)
    num_tokens += MESSAGE_PAD_TOKENS

    return num_tokens


def get_tokens_to_trigger_summary(context_window: int, completion_tokens: int) -> int:
    return min(int(0.9 * context_window), context_window - completion_tokens, TOKENS_TO_TRIGGER_SUMMARY)


def xml(input: str, tag: str) -> str:
    return f"<{tag}>{input}</{tag}>"


def get_root_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parent.parent


def get_embedding_model():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sentence_transformers.SentenceTransformer(EMBEDDING_MODEL, cache_folder=get_root_dir() / "cache")


def get_sentences(text: str, pattern=r"([^.!?]+)([.!?]+[\s]*)") -> list:
    matches = re.findall(pattern, text)
    return [m[0] + m[1] for m in matches]


def excise_middle_sentence(text: str, sep: str = " ") -> str:
    sentences = get_sentences(text)
    sentences.pop(len(sentences) // 2)
    return sep.join(sentences)
