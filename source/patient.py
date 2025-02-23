import logging
import os
import re
from typing import List, Tuple

import numpy as np
import openai
from tqdm import tqdm

from . import patient_utils

logger = logging.getLogger("ai-patient")

COMPLETION_TOKENS = int(os.getenv("COMPLETION_TOKENS", 500))
CONTEXT_WINDOW = int(os.getenv("CONTEXT_WINDOW", 4097))
STEPS_TO_REFLECTION = int(os.getenv("STEPS_TO_REFLECTION", 2))
TOP_RELEVANT_MEMORIES_TO_FETCH = int(os.getenv("TOP_RELEVANT_MEMORIES_TO_FETCH", 5))


class Patient:

    def __init__(self, persona: dict, prompts: dict, initial_conversation: List[dict], therapist_name: str):
        self.persona_id = persona["id"]
        self.persona_name = persona["name"]
        self.summary = persona["definition"]["summary"]
        self.personality = persona["definition"]["personality"]
        self.therapist_name = therapist_name

        self.prompts = prompts
        self.conversation = initial_conversation
        self.conversation_metadata = [{}] * len(initial_conversation)
        self.conversation_summary = None
        self.steps_to_reflection = STEPS_TO_REFLECTION
        self.topics = {}
        self.oldest_talk_turn_to_remember = 0
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

        logger.info("Populating memories ...")
        self.memories = {}
        for memory in tqdm(persona.get("memories", {})):
            patient_memory = {k: memory[k] for k in ("content", "embed")}
            patient_memory.update(memory["metadata"])
            patient_memory["embedding"] = self._get_embedding(memory["embed"])
            self.memories[memory["embed"]] = patient_memory

    def _rename_role(self, role: str) -> str:
        return self.therapist_name if role == "user" else "You"

    def _get_embedding(self, text: str) -> np.ndarray:
        embeddings = self.embedding_model.encode(text, show_progress_bar=False, normalize_embeddings=True)
        return embeddings.astype(np.float64)

    def _parse(self, text: str, substitutes: dict = None, pattern=r"\{([^}]+)\}") -> str:
        substitutes_base = {k.upper(): v for k, v in self.__dict__.items()}
        if substitutes is not None:
            substitutes_base.update(substitutes)

        def replace(match):
            key = match.group(1)
            return str(substitutes_base.get(key, match.group(0)))

        return re.sub(pattern, replace, text)

    def _get_top_memories(self, embedding_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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

    def _trim(self, messages: List[dict]) -> List[dict]:
        if patient_utils.get_messages_size(messages) + self.completion_tokens - self.context_window > 0:
            logger.warning("Emergency messages trimming triggered ...")
        while patient_utils.get_messages_size(messages) + self.completion_tokens - self.context_window > 0:
            logger.info(f"messages size: {patient_utils.get_messages_size(messages)}")

            # if there's a system prompt and dialog, remove dialog first
            if messages[0]["role"] == "system" and len(messages) > 1:
                if len(messages[1]["content"]) > 0:
                    messages[1]["content"] = patient_utils.excise_middle_sentence(messages[1]["content"])
                if len(messages[1]["content"]) == 0:
                    messages.pop(1)
            else:
                messages[0]["content"] = patient_utils.excise_middle_sentence(messages[0]["content"])
        return messages

    def _recall_conversation(self) -> List[dict]:
        old_conversation = [self.conversation[self.oldest_talk_turn_to_remember]]

        while (
            patient_utils.get_messages_size(old_conversation) < self.tokens_to_summarize
            and self.oldest_talk_turn_to_remember < len(self.conversation) - 2
        ):
            self.oldest_talk_turn_to_remember += 1
            old_conversation.append(self.conversation[self.oldest_talk_turn_to_remember])
            logger.debug(
                "Updated `oldest_talk_turn_to_remember` to: "
                f"{self.oldest_talk_turn_to_remember}/{len(self.conversation)}"
            )

        # Ensure the conversation ends with what the agent last said
        if old_conversation[-1]["role"] != "assistant":
            self.oldest_talk_turn_to_remember += 1
            old_conversation.append(self.conversation[self.oldest_talk_turn_to_remember])
            logger.debug(
                "Updated `oldest_talk_turn_to_remember` to: "
                f"{self.oldest_talk_turn_to_remember}/{len(self.conversation)}"
            )
        return old_conversation

    def _summary_messages(self) -> List[dict]:
        system_prompt = self.prompts["system_preamble"]

        if self.conversation_summary is not None:
            system_prompt += self.prompts["summarize"]["previous_summary"]

        system_prompt += self.prompts["summarize"]["command"]
        system_prompt = self._parse(
            system_prompt,
            {
                "OLD_CONVERSATION": "\n".join(
                    [
                        f"{self._rename_role(talk_turn['role'])}:\t{talk_turn['content']}"
                        for talk_turn in self._recall_conversation()
                    ]
                )
            },
        )

        messages = [{"role": "system", "content": system_prompt}]

        return messages

    def _llm_call(self, messages: List[dict]) -> str:
        logger.info(
            "Information sent to the LLM:"
            + "\n*** START LLM CALL ***\n\n"
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
            patient_utils.get_messages_size(self.conversation[self.oldest_talk_turn_to_remember :])
            > self.tokens_to_trigger_summary
        ) and (self.conversation_metadata[-1] != {}):
            logger.info("Summarization is triggered ... ")
            messages = self._summary_messages()
            messages = self._trim(messages)

            logger.info("This is an intermediate llm call for summarizing the old conversation:")
            self.conversation_summary = self._llm_call(messages)

            logger.info(f"Creating summary memory with following content: \n {self.conversation_summary}")
            self.memories[self.conversation_summary] = {
                "embed": self.conversation_summary,
                "embedding": self._get_embedding(self.conversation_summary),
                "content": self.conversation_summary,
                "is_summary": True,
                **self.conversation_metadata[-1],
            }

    def _get_mood(self, topics: np.ndarray, entailments: np.ndarray) -> float:
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

    def _topic_content(self, topic: str, mood: float) -> str:
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

    def _topical_system_prompt(self, topical_content: str, verbosity: int) -> str:
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

    def _get_verbosity(self, topic: str) -> int:
        # the system prompt will say to produce no more than 8-25 tokens, depending on perceived importance
        perception = self.memories[topic]["importance_belief"] + self.topics[topic]["importance_delta"]
        return int(np.round(perception * 27 + 8))

    def _topical_prompt(self, topics: np.ndarray, entailments: np.ndarray) -> Tuple[str, dict]:
        top_topic = topics[np.argmax(entailments)]
        valence_delta, importance_delta = self._update_topic_state(top_topic)

        mood = self._get_mood(topics, entailments)
        topic_content = self._topic_content(top_topic, mood)
        verbosity = self._get_verbosity(top_topic)
        system_prompt = self._topical_system_prompt(topic_content, verbosity)

        metadata = {
            "topic": top_topic,
            "mood": mood,
            "perceived_valence": self.memories[top_topic]["valence_belief"] + valence_delta,
            "valence_belief": self.memories[top_topic]["valence_belief"],
            "valence": self.memories[top_topic]["valence"],
            "perceived_importance": self.memories[top_topic]["importance_belief"] + importance_delta,
            "importance_belief": self.memories[top_topic]["importance_belief"],
            "importance": self.memories[top_topic]["importance"],
        }

        return system_prompt, metadata

    def _topic_messages(self) -> Tuple[List[dict], dict]:
        context = " ".join([talk_turn["content"] for talk_turn in self.conversation[-3:]])

        embedding_vector = self._get_embedding(context)

        topics, entailments = self._get_top_memories(embedding_vector)

        system_prompt, metadata = self._topical_prompt(topics, entailments)

        messages = [{"role": "system", "content": system_prompt}] + self.conversation[
            self.oldest_talk_turn_to_remember :
        ]

        return messages, metadata

    def _reflection_messages(self, top_n: int = 2) -> Tuple[List[dict], dict]:
        logger.info("Reflection is triggered ...")
        system_prompt = self._parse(self.prompts["system_preamble"]) + patient_utils.xml(
            self.personality, "personality"
        )

        if self.conversation_summary is not None:
            system_prompt += self._parse(self.prompts["summarize"]["previous_summary"])

        system_prompt = system_prompt + self._parse(self.prompts["reflect"]["preamble"])

        conversation_string = (
            "\n"
            + "\n".join(
                [
                    f"{self._rename_role(talk_turn['role'])}:\t{talk_turn['content']}"
                    for talk_turn in self.conversation[self.oldest_talk_turn_to_remember :]
                ]
            )
            + "\n"
        )

        system_prompt += self._parse(
            self.prompts["reflect"]["conversation"], {"CONVERSATION": conversation_string}
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

        system_prompt += patient_utils.xml(self._parse(self.prompts["reflect"]["final_command"],
                                     {"RESPONSE": self.conversation[-1]["content"]}), "command")
        messages = [{"role": "system", "content": system_prompt}]

        metadata = {"Topic": "None, the patient is reflecting on the conversation"}

        return messages, metadata

    def _check_to_reflect(self) -> bool:
        valence_deltas = np.array([topic["valence_delta"] for topic in self.topics.values()])
        importance_deltas = np.array([topic["importance_delta"] for topic in self.topics.values()])

        valence_cap = np.mean(
            [
                abs(float(self.memories[memory]["valence"]) - float(self.memories[memory]["valence_belief"]))
                for memory in self.memories
                if not self.memories[memory].get("is_summary", False)
            ]
        )
        importance_cap = np.mean(
            [
                float(self.memories[memory]["importance"]) - float(self.memories[memory]["importance_belief"])
                for memory in self.memories
                if not self.memories[memory].get("is_summary", False)
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

    def _response_messages(self) -> Tuple[List[dict], dict]:
        messages, metadata = self._reflection_messages() if self._check_to_reflect() else self._topic_messages()
        return messages, metadata

    def response(self, message: str) -> Tuple[str, List[dict], dict]:
        self.conversation.append({"role": "user", "content": message})

        self._summarize()

        messages, metadata = self._response_messages()

        messages = self._trim(messages)
        response = self._llm_call(messages)

        self.conversation.append({"role": "assistant", "content": response})
        self.conversation_metadata.append(metadata)

        return response, messages, metadata
