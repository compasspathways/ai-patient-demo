import os
import pathlib
from typing import List

MESSAGE_PAD_TOKENS = int(os.getenv("MESSAGE_PAD_TOKENS", 3))
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
    return min(int(0.9 * context_window), context_window - completion_tokens)


def xml(input: str, tag: str):
    return f"<{tag}>{input}</{tag}>"


def get_root_dir():
    return pathlib.Path(__file__).parent.parent.parent
