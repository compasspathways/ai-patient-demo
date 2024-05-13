import numpy as np
import openai
import sentence_transformers
from . import utils


class Patient:
    def __init__(self, id: str, persona: dict) -> None:
        self.id = id

        self.model: str = persona["model"]["name"]
        self.context_window = persona["model"]["context_window"]
        self.message_pad_tokens = persona["model"]["message_pad_tokens"]
        self.completion_tokens = persona["model"]["completion_tokens"]
        self.client = openai.OpenAI()
        self.embedding_model = sentence_transformers.SentenceTransformer(
            persona["embedding_model"], cache_folder=utils.get_root_dir() / "cache"
        )

        self.summary: str = persona["summary"]
        self.personality: str = persona["personality"]

        self.steps_to_reflection = 6
        self.intention_increment = 0.15

        self.conversation = []
        self.memories = {}
        self.states = {}

    def _get_embedding(self, text: str) -> np.ndarray:
        embeddings = self.embedding_model.encode(text, show_progress_bar=False, normalize_embeddings=True)
        return embeddings.astype(np.float64)

    def respond_to(self, message):
        self.conversation.append({"role": "user", "content": message})
        response = (
            self.client.chat.completions.create(model=self.model, messages=self.conversation)
            .choices[0]
            .message.content
        )
        self.conversation.append({"role": "assistant", "content": response})

        return response
