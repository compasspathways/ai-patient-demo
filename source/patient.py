import os
import re
from typing import List

import numpy as np
import openai
from tqdm import tqdm

from . import patient_utils

COMPLETION_TOKENS = int(os.getenv("COMPLETION_TOKENS", 500))
CONTEXT_WINDOW = int(os.getenv("CONTEXT_WINDOW", 4097))
STEPS_TO_REFLECTION = int(os.getenv("STEPS_TO_REFLECTION", 6))
TOP_RELEVANT_MEMORIES_TO_FETCH = int(os.getenv("TOP_RELEVANT_MEMORIES_TO_FETCH", 5))


class Patient:
    def __init__(self, persona: dict, prompts: dict, initial_conversation: List[dict]):
        self.persona_id = persona["id"]
        self.persona_name = persona["name"]
        self.summary = persona["definition"]["summary"]
        self.personality = persona["definition"]["personality"]

        self.prompts = prompts
        self.conversation = initial_conversation
        self.conversation_summary = None
        self.steps_to_reflection = STEPS_TO_REFLECTION
        self.topics = {}
        self.conversation_lines_forgotten = 0
        self.num_valence_reflections = 0
        self.num_importance_reflections = 0

        self.model_name = persona["model_id"]
        self.context_window = CONTEXT_WINDOW
        self.completion_tokens = COMPLETION_TOKENS
        self.api_client = openai.OpenAI()
        self.embedding_model = patient_utils.get_embedding_model()

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

    def _get_embedding(self, text: str) -> np.ndarray:
        embeddings = self.embedding_model.encode(text, show_progress_bar=False, normalize_embeddings=True)
        return embeddings.astype(np.float64)

    def _insert_memory(self, text: str):
        self.memories[text] = {
            "embed": text,
            "embedding": self._get_embedding(text),
            "content": text,
            "metadata": {},
        }

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

    def _get_top_memories(self, embedding_vector: np.ndarray) -> tuple:
        embeddings = np.array([memory["embedding"] for memory in self.memories.values()])
        distances = np.linalg.norm(np.array(embeddings) - np.array(embedding_vector), axis=1)

        idxs = np.argpartition(distances, TOP_RELEVANT_MEMORIES_TO_FETCH)[:TOP_RELEVANT_MEMORIES_TO_FETCH]

        top_memories_keys = np.array(list(self.memories.keys()))[idxs]
        top_memories = [
            {**self.memories[key], "distance": distance}
            for key, distance in zip(top_memories_keys, distances[idxs])
        ]

        memory_contents = np.array([memory["embed"] for memory in top_memories])
        entailments = 2 - np.array([memory["distance"] for memory in top_memories])

        return memory_contents, entailments

    def _trim(self, messages: List[dict]):
        if patient_utils.get_messages_size(messages) + self.completion_tokens - self.context_window > 0:
            print("Emergency messages trimming triggered.")
        while patient_utils.get_messages_size(messages) + self.completion_tokens - self.context_window > 0:
            print(patient_utils.get_messages_size(messages))

            # if there's a system prompt and dialog, remove dialog first
            if messages[0]["role"] == "system" and len(messages) > 1:
                if len(messages[1]["content"]) > 0:
                    messages[1]["content"] = patient_utils.excise_middle_sentence(messages[1]["content"])
                if len(messages[1]["content"]) == 0:
                    messages.pop(1)
            else:
                messages[0]["content"] = patient_utils.excise_middle_sentence(messages[0]["content"])
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

    def _summarize(self):
        if (
            patient_utils.get_messages_size(self.conversation[self.conversation_lines_forgotten :])
            > self.tokens_to_trigger_summary
        ):
            messages = self._summary_messages()
            messages = self._trim(messages)

            self.conversation_summary = self._llm_call(messages)

            self._insert_memory(
                self.conversation_summary,
                type="summary",
            )

    def _get_mood(self, topics, entailments):
        valence_beliefs, importance_beliefs = [], []
        for topic in topics:
            valence = self.memories[topic]["valence_belief"]
            importance = self.memories[topic]["importance_belief"]
            if topic in self.topics:
                valence += self.topics[topic]["valence_delta"]
                importance += self.topics[topic]["importance_delta"]
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
            + self.prompts["state_descriptions"]["importance_descriptions"][importance_likert]
            + self._parse(
                self.prompts["memories_topic"]["mood"],
                {"MOOD": self.prompts["state_descriptions"]["valence_descriptions"][mood_likert]},
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

    def _update_topic_state(self, topic: str):
        if topic not in self.topics:
            self.topics[topic] = {}

        valence_delta = 0.5 * (
            self.topics[topic].get("valence_delta", 0)
            + self.memories[topic]["valence"]
            - self.memories[topic]["valence_belief"]
        )

        importance_delta = 0.5 * (
            self.topics[topic].get("importance_delta", 0)
            + self.memories[topic]["importance"]
            - self.memories[topic]["importance_belief"]
        )

        self.topics[topic].update({"importance_delta": importance_delta, "valence_delta": valence_delta})

        return valence_delta, importance_delta

    def _get_verbosity(self, topic):
        # the system prompt will say to produce no more than 8-25 tokens, depending on perceived importance
        perception = self.memories[topic]["importance_belief"] + self.topics[topic]["importance_delta"]
        return int(np.round(perception * 27 + 8))

    def _topical_prompt(self, topics, entailments):
        top_topic = topics[np.argmax(entailments)]
        valence_delta, importance_delta = self._update_topic_state(top_topic)

        mood = self._get_mood(topics, entailments)
        topic_content = self._topic_content(top_topic, mood)
        verbosity = self._get_verbosity(top_topic)
        system_prompt = self._topical_system_prompt(topic_content, verbosity)

        metadata = {
            "topic": top_topic,
            "General Mood": mood,
            "Perceived Valence": self.memories[top_topic]["valence_belief"] + valence_delta,
            "True Valence": self.memories[top_topic]["valence_belief"],
            "Perceived Importance": self.memories[top_topic]["importance_belief"] + importance_delta,
            "True Importance": self.memories[top_topic]["importance_belief"],
        }

        return system_prompt, metadata

    def _topic_messages(self):
        embedding_vector = self._get_embedding(self.conversation[-1]["content"])
        topics, entailments = self._get_top_memories(embedding_vector)

        system_prompt, metadata = self._topical_prompt(topics, entailments)

        messages = [{"role": "system", "content": system_prompt}] + self.conversation[
            self.conversation_lines_forgotten :
        ]

        return messages, metadata

    def _reflection_messages(self, top_n: int = 2):
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
                    self.topics.items(),
                    key=lambda item: abs(item[1]["valence_delta"]),
                    reverse=True,
                )[:top_n]
            }
            topics = topics.union(best_valence_topics)

        if self.reflect_importance:
            best_importance_topics = {
                i[0]
                for i in sorted(
                    self.topics.items(),
                    key=lambda item: abs(item[1]["importance_delta"]),
                    reverse=True,
                )[:top_n]
            }
            topics = topics.union(best_importance_topics)

        for topic in topics:
            valence_score = np.round(
                float(self.memories[topic]["valence_belief"]) + self.topics[topic]["valence_delta"],
                2,
            )
            importance_score = np.round(
                float(self.memories[topic]["importance_belief"]) + self.topics[topic]["importance_delta"],
                2,
            )
            system_prompt = (
                system_prompt
                + f"Topic:{topic};"
                + self.memories[topic]["content"]
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
        valence_deltas = np.array([topic["valence_delta"] for topic in self.topics.values()])
        importance_deltas = np.array([topic["importance_delta"] for topic in self.topics.values()])

        valence_cap = np.mean(
            [
                abs(float(self.memories[memory]["valence"]) - float(self.memories[memory]["valence_belief"]))
                for memory in self.memories
            ]
        )
        importance_cap = np.mean(
            [
                float(self.memories[memory]["importance"]) - float(self.memories[memory]["importance_belief"])
                for memory in self.memories
            ]
        )

        self.reflect_valence = sum(np.abs(valence_deltas)) > (valence_cap * self.steps_to_reflection * 2) * (
            self.num_valence_reflections + 1
        )
        self.reflect_importance = sum(importance_deltas) > (importance_cap * self.steps_to_reflection * 2) * (
            self.num_importance_reflections + 1
        )

        if self.reflect_valence:
            self.num_valence_reflections += 1

        if self.reflect_importance:
            self.num_importance_reflections += 1

        return self.reflect_valence or self.reflect_importance

    def _response_messages(self):
        messages, metadata = self._reflection_messages() if self._check_to_reflect() else self._topic_messages()
        return messages, metadata

    def response(self, message):
        self.conversation.append({"role": "user", "content": message})
        self._summarize()
        messages, metadata = self._response_messages()
        messages = self._trim(messages)
        response = self._llm_call(messages)
        self.conversation.append({"role": "assistant", "content": response})

        return response, messages, metadata
