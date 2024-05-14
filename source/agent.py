import copy
import json
import os
import re
import time
from typing import List

import numpy as np
import openai
import sentence_transformers

from . import utils

DEFAULT_STATE_VARS = {
    "num_valence_reflections": {"type": int, "value": 0},
    "num_importance_reflections": {"type": int, "value": 0},
    "intention_set": {"type": float, "value": 0},
}

COMPLETION_TOKENS = int(os.getenv("COMPLETION_TOKENS", 500))
CONTEXT_WINDOW = int(os.getenv("CONTEXT_WINDOW", 4097))
MESSAGE_PAD_TOKENS = int(os.getenv("MESSAGE_PAD_TOKENS", 3))


def _get_tokens_to_trigger_summary(context_window: int, completion_tokens: int) -> int:
    return min(int(0.9 * context_window), context_window - completion_tokens)


class BaseAgent:

    def __init__(
        self,
        dbm,
        persona: dict,
        conversation_metadata: dict,
        conversation: list,
    ):
        self.dbm = dbm

        # Persona params
        self.persona_id = persona["id"]
        self.persona_name = persona["name"]

        # Conversation Params
        self.conversation_id = conversation_metadata["id"]
        self.user_id = conversation_metadata["user_id"]
        self.user_name = conversation_metadata["user_name"]

        self.conversation = conversation

        # Prompts
        self.prompts: dict = persona["prompts"]

        # Model params
        self.model_name = persona["model_name"]
        self.context_window = CONTEXT_WINDOW
        self.message_pad_tokens = MESSAGE_PAD_TOKENS
        self.completion_tokens = COMPLETION_TOKENS

        self.api_client = openai.OpenAI()
        self.embedding_model = sentence_transformers.SentenceTransformer(
            persona["embedding_model"], cache_folder=utils.get_root_dir() / "cache"
        )

        self._set_state()
        self._set_conversation_lines_forgotten()
        self._set_summary_params(persona)
        self.conversation_summary = None

        summaries = self.dbm.get_memories(
            where_clause=f"""
                persona_id='{self.persona_id}' AND conversation_id='{self.conversation_id}' AND type='summary'
            """
        )
        if summaries:
            self.conversation_summary = summaries[-1]["content"]

    def _set_state(self):
        self.state = {}
        for state in self.dbm.get_rows("state", self.conversation_id):
            if state["type"] not in self.state:
                self.state[state["type"]] = {}
            if state["topic"] not in self.state[state["type"]]:
                self.state[state["type"]][state["topic"]] = {}
            self.state[state["type"]][state["topic"]][state["key"]] = state["value"]

    def _set_conversation_lines_forgotten(self):
        if (
            "general" in self.state
            and "memory" in self.state["general"]
            and "conversation_lines_forgotten" in self.state["general"]["memory"]
        ):
            self.conversation_lines_forgotten = self.state["general"]["memory"]["conversation_lines_forgotten"]
        else:
            print("`conversation_lines_forgotten` is not in conversation state data; using default of 0!")
            self.conversation_lines_forgotten = 0

    def _set_summary_params(self, persona: dict):
        self.tokens_to_trigger_summary = persona.get("tokens_to_trigger_summary")
        if self.tokens_to_trigger_summary is None:
            print("`tokens_to_trigger_summary` is not in persona data; using default!")
            self.tokens_to_trigger_summary = _get_tokens_to_trigger_summary(
                self.context_window, self.completion_tokens
            )

        self.tokens_to_summarize = persona.get("tokens_to_summarize")
        if self.tokens_to_summarize is None:
            print("`tokens_to_summarize` is not in persona data; using default!")
            self.tokens_to_summarize = int(0.4 * self.context_window)

        self.tokens_for_summary = persona.get("tokens_for_summary")
        if self.tokens_for_summary is None:
            print("`tokens_for_summary` is not in persona data; using default!")
            self.tokens_to_summarize = int(0.1 * self.context_window)

    @staticmethod
    def _xml(string, tag):
        return f"""<{tag}>{string}</{tag}>"""

    def _parse(self, string, subs=None, pattern=r"\{([^}]+)\}"):
        subs_base = {k.upper(): v for k, v in self.__dict__.items()}
        if subs is None:
            subs = subs_base
        else:
            subs.update(subs_base)

        def replace(match):
            key = match.group(1)
            return str(subs.get(key, match.group(0)))

        return re.sub(pattern, replace, string)

    def _messages_size(self, messages: List[dict]) -> int:
        # TODO
        return self.api_client.get_messages_size(messages, self.message_pad_tokens)

    def _get_sentences(self, string: str, pattern=r"([^.!?]+)([.!?]+[\s]*)") -> list:
        matches = re.findall(pattern, string)
        return [m[0] + m[1] for m in matches]

    def _excise_middle_sentence(self, string: str, sep: str = " "):
        sentences = self._get_sentences(string)
        sentences.pop(len(sentences) // 2)
        return sep.join(sentences)

    def _trim(self, messages: List[dict]):
        if self._messages_size(messages) + self.completion_tokens - self.context_window > 0:
            print("Emergency messages trimming triggered.")
        while self._messages_size(messages) + self.completion_tokens - self.context_window > 0:
            print(self._messages_size(messages))

            # if there's a system prompt and dialog, remove dialog first
            if messages[0]["role"] == "system" and len(messages) > 1:
                if len(messages[1]["content"]) > 0:
                    messages[1]["content"] = self._excise_middle_sentence(messages[1]["content"])
                if len(messages[1]["content"]) == 0:
                    messages.pop(1)
            else:
                messages[0]["content"] = self._excise_middle_sentence(messages[0]["content"])

    def _recall_conversation(self):
        old_conversation = [self.conversation[self.conversation_lines_forgotten]]

        while (
            self._messages_size(old_conversation) < self.tokens_to_summarize
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

    def _summarize(self):
        if (
            self._messages_size(self.conversation[self.conversation_lines_forgotten :])
            > self.tokens_to_trigger_summary
        ):
            messages = self._summary_messages()
            self._trim(messages)

            self.conversation_summary = self._llm_call(messages)

            self.dbm.insert_memory(
                self.persona_id,
                self.conversation_summary,
                conversation_id=self.conversation_id,
                user_id=self.user_id,
                type="summary",
            )

            self.dbm.update_state(
                self.conversation_id,
                "general",
                "memory",
                "conversation_lines_forgotten",
                self.conversation_lines_forgotten,
            )

    def _llm_call(self, messages: List[dict]) -> str:
        print(
            "\n*** START LLM CALL ***\n"
            + "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            + "\n*** END LLM CALL ***\n"
        )
        return (
            self.api_client.chat.completions.create(model=self.model_name, messages=self.conversation)
            .choices[0]
            .message.content
        )

    def _make_messages(self, system_prompt: str) -> List[dict]:
        messages = [{"role": "system", "content": system_prompt}] + self.conversation[
            self.conversation_lines_forgotten :
        ]
        return messages

    def response(self):
        self._summarize()
        messages, metadata = self._response_messages()
        self._trim(messages)

        return self._llm_call(messages), messages, metadata

    def append_conversation(self, speech: str):
        self.conversation.append({"role": "user", "content": speech})

    def update_speech_table(self, call_summary: dict):
        self.dbm.insert(
            "speech_acts",
            {
                "conversation_id": self.conversation_id,
                "role": "user",
                "content": call_summary["speech"],
                "metadata": '{"source": "human_input"}',
            },
        )
        self.dbm.insert(
            "speech_acts",
            {
                "conversation_id": self.conversation_id,
                "role": "assistant",
                "emotion": call_summary["emotion"],
                "content": call_summary["reply"],
                "metadata": call_summary["display_data"],
            },
        )


class Patient(BaseAgent):
    def __init__(self, dbm, persona, conv_data, conversation):
        super().__init__(dbm, persona, conv_data, conversation)

        self.steps_to_reflection = 6
        self.summary = persona.get("summary", self.prompts["summary"])
        self.personality = persona.get("personality", self.prompts["summary"])
        self.intention_increment = 0.15

        self.memories = {}
        memories = persona.get("memories", {})
        for mem in memories:
            patient_mem = {k: mem[k] for k in ("content", "type")}
            patient_mem.update(mem["metadata"])
            self.memories[mem["embed_text"]] = copy.deepcopy(patient_mem)

        if not self.state:
            self.state = {k: v["type"](v["value"]) for k, v in DEFAULT_STATE_VARS.items()}
            self.state["topic"] = {}

            for var in DEFAULT_STATE_VARS:
                self.dbm.insert(
                    "state",
                    {
                        "conversation_id": self.conversation_id,
                        "type": "general",
                        "topic": "memory",
                        "key": var,
                        "value": json.dumps(DEFAULT_STATE_VARS[var]["value"]),
                    },
                )
        else:
            self.state.update(
                {k: v["type"](self.state["general"]["memory"][k]) for k, v in DEFAULT_STATE_VARS.items()}
            )

            if "topic" not in self.state:
                self.state["topic"] = {}

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
            self._xml(self.prompts["system_preamble"], "preamble")
            + self._xml(self.summary, "bio")
            + self._xml(self.personality, "personality")
        )

        if self.conversation_summary is not None:
            system_prompt += self.prompts["summarize"]["previous_summary"]

        system_prompt += self._xml(topical_content, "thought")
        system_prompt += self._xml(self.prompts["stance"], "stance")
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
                self.dbm.insert(
                    "state",
                    {
                        "conversation_id": self.conversation_id,
                        "type": "topic",
                        "topic": topic,
                        "key": key,
                        "value": json.dumps(value),
                    },
                )
                self.state["topic"][topic][key] = value
        else:
            self.dbm.update_state(
                self.conversation_id,
                "topic",
                topic,
                "importance_delta",
                json.dumps(importance_delta),
            )

            self.dbm.update_state(
                self.conversation_id,
                "topic",
                topic,
                "valence_delta",
                json.dumps(valence_delta),
            )

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
            self._xml(self._parse(self.prompts["system_preamble"]), "preamble")
            + self._xml(self.summary, "bio")
            + self._xml(self.personality, "personality")
        )

        if self.conversation_summary is not None:
            system_prompt += self._parse(self.prompts["summarize"]["previous_summary"])

        intention_likert = min(3, int(np.floor(self.state["intention_set"] * 4)))
        system_prompt += self._parse(
            self.prompts["intention_setting_ability"][str(intention_likert)],
            {"INTENTION": self.memories[intention]["content"]},
        )
        system_prompt += self._xml(self._parse(self.prompts["intention_stance"]), "stance")

        self._update_intention_state()

        metadata = {}

        return system_prompt, metadata

    def _update_intention_state(self):
        self.state["intention_set"] += self.intention_increment
        self.state["intention_set"] = min(1, self.state["intention_set"])
        self.dbm.update_state(
            self.conversation_id,
            "general",
            "memory",
            "intention_set",
            json.dumps(self.state["intention_set"]),
        )

    def _topic_messages(self):
        start_secs = time.time()
        types, contents, entailments = self.dbm.get_dists(
            self.conversation[-1]["content"], persona_id=self.persona_id
        )

        print(f"get_dists: {time.time() - start_secs} seconds.")
        print(
            {
                "types (first 5)": types[:5],
                "contents (first 5)": contents[:5],
                "entailments (first 5)": entailments[:5],
            }
        )

        type = types[np.argmax(entailments)]
        contents = contents[types == type]
        entailments = entailments[types == type]

        if type == "topic":
            system_prompt, metadata = self._topical_prompt(contents, entailments)
            print(f"_topical_system_prompt: {time.time() - start_secs} seconds.")

        elif type == "intention":
            system_prompt, metadata = self._intention_prompt(contents, entailments)
            print(f"intention_system_prompt: {time.time() - start_secs} seconds.")

        messages = self._make_messages(system_prompt)

        return messages, metadata

    def _reflection_messages(self, n=2):
        system_prompt = self._parse(self.prompts["system_preamble"]) + self._xml(self.personality, "personality")

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

        if self.reflect_valence:
            self.dbm.update_state(
                self.conversation_id,
                "general",
                "memory",
                "num_valence_reflections",
                json.dumps(self.state["num_valence_reflections"] + 1),
            )
        if self.reflect_importance:
            self.dbm.update_state(
                self.conversation_id,
                "general",
                "memory",
                "num_importance_reflections",
                json.dumps(self.state["num_importance_reflections"] + 1),
            )

        return self.reflect_valence or self.reflect_importance

    def _response_messages(self):
        reflect = self._check_to_reflect()
        messages, metadata = self._reflection_messages() if reflect else self._topic_messages()
        return messages, metadata
