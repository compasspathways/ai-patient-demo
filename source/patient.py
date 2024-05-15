import os
import re
import warnings
from typing import List

import numpy as np
import openai
import sentence_transformers
from tqdm import tqdm

from . import patient_utils

DEFAULT_STATE = {
    "num_valence_reflections": 0,
    "num_importance_reflections": 0,
    "intention_set": 0.0,
    "topic": {},
}

COMPLETION_TOKENS = int(os.getenv("COMPLETION_TOKENS", 500))
CONTEXT_WINDOW = int(os.getenv("CONTEXT_WINDOW", 4097))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
CACHE_DIR = patient_utils.get_root_dir() / "cache"

STEPS_TO_REFLECTION = int(os.getenv("STEPS_TO_REFLECTION", 6))
INTENTION_INCREMENT = float(os.getenv("INTENTION_INCREMENT", 0.15))

TOP_RELEVANT_MEMORIES_TO_FETCH = int(os.getenv("TOP_RELEVANT_MEMORIES_TO_FETCH", 5))


class Patient:
    def __init__(self, persona: dict, conversation: List[dict], prompts: dict):
        self.persona_id = persona["id"]
        self.persona_name = persona["name"]
        self.summary = persona["definition"]["summary"]
        self.personality = persona["definition"]["personality"]

        self.prompts = prompts
        self.conversation = conversation
        self.conversation_summary = None
        self.steps_to_reflection = STEPS_TO_REFLECTION
        self.intention_increment = INTENTION_INCREMENT
        self.state = DEFAULT_STATE
        self.conversation_lines_forgotten = 0

        self.model_name = persona["model_id"]
        self.context_window = CONTEXT_WINDOW
        self.completion_tokens = COMPLETION_TOKENS
        self.api_client = openai.OpenAI()
        self._set_embedding_model()

        self.tokens_to_summarize = int(0.4 * self.context_window)
        self.tokens_for_summary = int(0.1 * self.context_window)
        self.tokens_to_trigger_summary = patient_utils.get_tokens_to_trigger_summary(
            self.context_window, self.completion_tokens
        )

        print(">>> Populating memories ...")
        self.memories = {}
        for memory in tqdm(persona.get("memories", {})):
            patient_memory = {k: memory[k] for k in ("content", "embed")}
            patient_memory.update(memory["metadata"])
            patient_memory["embedding"] = self._get_embedding(memory["embed"])
            self.memories[memory["embed"]] = patient_memory

    def _set_embedding_model(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.embedding_model = sentence_transformers.SentenceTransformer(
                EMBEDDING_MODEL, cache_folder=CACHE_DIR
            )

    def _get_embedding(self, text: str) -> np.ndarray:
        embeddings = self.embedding_model.encode(text, show_progress_bar=False, normalize_embeddings=True)
        return embeddings.astype(np.float64)

    def get_top_memories(self, embedding_vector: np.ndarray) -> tuple:
        embeddings = np.array([memory["embedding"] for memory in self.memories])
        distances = np.linalg.norm(np.array(embeddings) - np.array(embedding_vector), axis=1)

        idxs = np.argpartition(distances, TOP_RELEVANT_MEMORIES_TO_FETCH)[:TOP_RELEVANT_MEMORIES_TO_FETCH]

        top_memories_keys = np.array(self.memories.keys())[idxs]
        top_memories = [self.memories[key] for key in top_memories_keys]

        types = np.array([memory["type"] for memory in top_memories])
        contents = np.array([memory["embed_text"] for memory in top_memories])
        entailments = 2 - np.array([memory["distance"] for memory in top_memories])

        return types, contents, entailments

    def _parse(self, text: str, subs: dict = None, pattern=r"\{([^}]+)\}"):
        subs_base = {k.upper(): v for k, v in self.__dict__.items()}
        if subs is None:
            subs = subs_base
        else:
            subs.update(subs_base)

        def replace(match):
            key = match.group(1)
            return str(subs.get(key, match.group(0)))

        return re.sub(pattern, replace, text)

    def _get_sentences(self, text: str, pattern=r"([^.!?]+)([.!?]+[\s]*)") -> list:
        matches = re.findall(pattern, text)
        return [m[0] + m[1] for m in matches]

    def _excise_middle_sentence(self, text: str, sep: str = " "):
        sentences = self._get_sentences(text)
        sentences.pop(len(sentences) // 2)
        return sep.join(sentences)

    def _trim(self, messages: List[dict]):
        if patient_utils.get_messages_size(messages) + self.completion_tokens - self.context_window > 0:
            print("Emergency messages trimming triggered.")
        while patient_utils.get_messages_size(messages) + self.completion_tokens - self.context_window > 0:
            print(patient_utils.get_messages_size(messages))

            # if there's a system prompt and dialog, remove dialog first
            if messages[0]["role"] == "system" and len(messages) > 1:
                if len(messages[1]["content"]) > 0:
                    messages[1]["content"] = self._excise_middle_sentence(messages[1]["content"])
                if len(messages[1]["content"]) == 0:
                    messages.pop(1)
            else:
                messages[0]["content"] = self._excise_middle_sentence(messages[0]["content"])
        return messages

    def _recall_conversation(self):
        old_conversation = [self.conversation[self.conversation_lines_forgotten]]

        while (
            patient_utils.get_messages_size(old_conversation) < self.tokens_to_summarize
            and self.conversation_lines_forgotten < len(self.conversation) - 2
        ):
            self.conversation_lines_forgotten += 1
            old_conversation.append(self.conversation[self.conversation_lines_forgotten])

        # ensure the conversation ends with what the agent last said
        if old_conversation[-1]["role"] != "assistant":
            self.conversation_lines_forgotten += 1
            old_conversation.append(self.conversation[self.conversation_lines_forgotten])
        return old_conversation

    def _summary_messages(self):
        system_prompt = self.prompts["system_preamble"]

        if self.conversation_summary is not None:
            system_prompt += self.prompts["summarize"]["previous_summary"]

        old_conversation = self._recall_conversation()
        oc_string = "\n".join([f"{t['actor'].upper()}:\t{t['content']}" for t in old_conversation])

        system_prompt += self.prompts["summarize"]["command"]
        system_prompt = self._parse(system_prompt, {"OLD_CONVERSATION": oc_string})

        messages = [{"role": "system", "content": system_prompt}]

        return messages

    def _insert_memory():
        pass

    def _summarize(self):
        if (
            patient_utils.get_messages_size(self.conversation[self.conversation_lines_forgotten :])
            > self.tokens_to_trigger_summary
        ):
            messages = self._summary_messages()
            messages = self._trim(messages)

            self.conversation_summary = self._llm_call(messages)

            self._insert_memory(
                self.persona_id,
                self.conversation_summary,
                type="summary",
            )

    def _llm_call(self, messages: List[dict]) -> str:
        print(
            "\n*** START LLM CALL ***\n"
            + "\n".join([f"{message['role']}: {message['content']}" for message in messages])
            + "\n*** END LLM CALL ***\n"
        )
        return (
            self.api_client.chat.completions.create(model=self.model_name, messages=messages)
            .choices[0]
            .message.content
        )

    def _get_mood(self, topics, entailments):
        valence_beliefs, importance_beliefs = [], []
        for topic in topics:
            valence = self.memories[topic]["valence_belief"]
            importance = self.memories[topic]["importance_belief"]
            if topic in self.state["topic"]:
                valence += self.state["topic"][topic]["valence_delta"]
                importance += self.state["topic"][topic]["importance_delta"]
            valence_beliefs.append(valence)
            importance_beliefs.append(importance)

        valence_beliefs = np.array(valence_beliefs)
        importance_beliefs = np.array(importance_beliefs)

        saliences = importance_beliefs * np.exp(entailments)

        mood = np.mean(valence_beliefs * saliences) / np.mean(saliences)

        return mood

    def _topic_content(self, topic, mood):
        mood_likert = np.round((mood + 1) * 5).astype(int)
        importance_likert = np.round(self.memories[topic]["importance_belief"] * 5).astype(int)

        topic_content = (
            self._parse(
                self.prompts["memories_topic"]["preamble"],
                {"CONTENT": self.memories[topic]["content"]},
            )
            + self.prompts["state_descriptions"]["importance_descriptions"][str(importance_likert)]
            + self._parse(
                self.prompts["memories_topic"]["mood"],
                {"MOOD": self.prompts["state_descriptions"]["valence_descriptions"][str(mood_likert)]},
            )
        )

        return topic_content

    def _topical_system_prompt(self, topical_content, verbosity):
        system_prompt = (
            patient_utils.xml(self.prompts["system_preamble"], "preamble")
            + patient_utils.xml(self.summary, "bio")
            + patient_utils.xml(self.personality, "personality")
        )

        if self.conversation_summary is not None:
            system_prompt += self.prompts["summarize"]["previous_summary"]

        system_prompt += patient_utils.xml(topical_content, "thought")
        system_prompt += patient_utils.xml(self.prompts["stance"], "stance")
        system_prompt = self._parse(system_prompt, {"VERBOSITY": verbosity})

        return system_prompt

    def _update_topic_state(self, topic):
        new_topic = topic not in self.state["topic"]
        valence_delta = 0.5 * (
            (self.state["topic"][topic]["valence_delta"] if not new_topic else 0)
            + self.memories[topic]["valence"]
            - self.memories[topic]["valence_belief"]
        )

        importance_delta = 0.5 * (
            (self.state["topic"][topic]["importance_delta"] if not new_topic else 0)
            + self.memories[topic]["importance"]
            - self.memories[topic]["importance_belief"]
        )

        if new_topic:
            self.state["topic"][topic] = {}
            for key, value in (
                ("importance_delta", importance_delta),
                ("valence_delta", valence_delta),
            ):
                self.state["topic"][topic][key] = value

        return valence_delta, importance_delta

    def _get_verbosity(self, topic):
        # the system prompt will say to produce no more than 8-25 tokens, depending on perceived importance
        perception = self.memories[topic]["importance_belief"] + self.state["topic"][topic]["importance_delta"]
        return int(np.round(perception * 27 + 8))

    def _topical_prompt(self, topics, entailments):
        topic = topics[np.argmax(entailments)]

        valence_delta, importance_delta = self._update_topic_state(topic)
        mood = self._get_mood(topics, entailments)
        topic_content = self._topic_content(topic, mood)
        verbosity = self._get_verbosity(topic)
        system_prompt = self._topical_system_prompt(topic_content, verbosity)
        metadata = {
            "topic": topic,
            "General Mood": mood,
            "Perceived Valence": self.memories[topic]["valence_belief"] + valence_delta,
            "True Valence": self.memories[topic]["valence_belief"],
            "Perceived Importance": self.memories[topic]["importance_belief"] + importance_delta,
            "True Importance": self.memories[topic]["importance_belief"],
        }

        return system_prompt, metadata

    def _intention_prompt(self, intentions, entailments):
        intention = intentions[np.argmax(entailments)]

        system_prompt = (
            patient_utils.xml(self._parse(self.prompts["system_preamble"]), "preamble")
            + patient_utils.xml(self.summary, "bio")
            + patient_utils.xml(self.personality, "personality")
        )

        if self.conversation_summary is not None:
            system_prompt += self._parse(self.prompts["summarize"]["previous_summary"])

        intention_likert = min(3, int(np.floor(self.state["intention_set"] * 4)))
        system_prompt += self._parse(
            self.prompts["intention_setting_ability"][str(intention_likert)],
            {"INTENTION": self.memories[intention]["content"]},
        )
        system_prompt += patient_utils.xml(self._parse(self.prompts["intention_stance"]), "stance")
        self._update_intention_state()

        return system_prompt, {}

    def _update_intention_state(self):
        self.state["intention_set"] += self.intention_increment
        self.state["intention_set"] = min(1, self.state["intention_set"])

    def _topic_messages(self):
        embedding_vector = self._get_embedding(self.conversation[-1]["content"])
        types, contents, entailments = self.get_top_memories(embedding_vector)

        type = types[np.argmax(entailments)]
        contents = contents[types == type]
        entailments = entailments[types == type]

        if type == "topic":
            system_prompt, metadata = self._topical_prompt(contents, entailments)
        elif type == "intention":
            system_prompt, metadata = self._intention_prompt(contents, entailments)

        messages = [{"role": "system", "content": system_prompt}] + self.conversation[
            self.conversation_lines_forgotten :
        ]

        return messages, metadata

    def _reflection_messages(self, n=2):
        system_prompt = self._parse(self.prompts["system_preamble"]) + patient_utils.xml(
            self.personality, "personality"
        )

        if self.conversation_summary is not None:
            system_prompt += self._parse(self.prompts["summarize"]["previous_summary"])

        system_prompt = system_prompt + self._parse(self.prompts["reflect"]["preamble"])

        conversation_string = (
            "\n".join(
                [
                    f"{t['actor'].upper()}:\t{t['content']}"
                    for t in self.conversation[self.conversation_lines_forgotten :]
                ]
            )
            + "\n"
        )

        system_prompt += self._parse(
            self.prompts["reflect"]["conversation"],
            {"CONVERSATION": conversation_string},
        )

        system_prompt += self.prompts["reflect"]["topics"]

        topics = set()
        if self.reflect_valence:
            best_valence_topics = {
                i[0]
                for i in sorted(
                    self.state["topic"].items(),
                    key=lambda item: abs(item[1]["valence_delta"]),
                    reverse=True,
                )[:n]
            }
            topics = topics.union(best_valence_topics)

        if self.reflect_importance:
            best_importance_topics = {
                i[0]
                for i in sorted(
                    self.state["topic"].items(),
                    key=lambda item: abs(item[1]["importance_delta"]),
                    reverse=True,
                )[:n]
            }
            topics = topics.union(best_importance_topics)

        for t in topics:
            valence_score = np.round(
                float(self.memories[t]["valence_belief"]) + self.state["topic"][t]["valence_delta"],
                2,
            )
            importance_score = np.round(
                float(self.memories[t]["importance_belief"]) + self.state["topic"][t]["importance_delta"],
                2,
            )
            system_prompt = (
                system_prompt
                + f"Topic:{t};"
                + self.memories[t]["content"]
                + self._parse(
                    self.prompts["reflect"]["scores"],
                    {
                        "VALENCE_SCORE": valence_score,
                        "IMPORTANCE_SCORE": importance_score,
                    },
                )
            )

        system_prompt += self._parse(self.prompts["reflect"]["final_command"])

        messages = [{"role": "system", "content": system_prompt}]

        metadata = {"Topic": "None, the patient is reflecting on the conversation"}

        return messages, metadata

    def _check_to_reflect(self):
        valence_deltas = np.array([v["valence_delta"] for v in self.state["topic"].values()])
        importance_deltas = np.array([v["importance_delta"] for v in self.state["topic"].values()])
        valence_cap = np.mean(
            [
                abs(float(self.memories[t]["valence"]) - float(self.memories[t]["valence_belief"]))
                for t in self.memories
                if self.memories[t]["type"] == "topic"
            ]
        )
        importance_cap = np.mean(
            [
                float(self.memories[t]["importance"]) - float(self.memories[t]["importance_belief"])
                for t in self.memories
                if self.memories[t]["type"] == "topic"
            ]
        )

        self.reflect_valence = sum(np.abs(valence_deltas)) > (valence_cap * self.steps_to_reflection * 2) * (
            self.state["num_valence_reflections"] + 1
        )
        self.reflect_importance = sum(importance_deltas) > (importance_cap * self.steps_to_reflection * 2) * (
            self.state["num_importance_reflections"] + 1
        )

        return self.reflect_valence or self.reflect_importance

    def _response_messages(self):
        messages, metadata = self._reflection_messages() if self._check_to_reflect() else self._topic_messages()
        return messages, metadata

    def response(self, message):
        self.conversation.append({"role": "user", "content": message})
        self._summarize()
        messages, metadata = self._response_messages()
        messages = self._trim(messages)

        return self._llm_call(messages), messages, metadata
