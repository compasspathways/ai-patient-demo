import os
import pathlib
import re
import warnings
from types import MethodType

import numpy as np
import torch
import yaml
from scipy.special import expit, logit
from transformers import pipeline


def get_root_dir():
    return pathlib.Path(__file__).parent.parent


def get_classifier():
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device {device} for inference.")

    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=device,
    )

    classifier.save_pretrained(get_root_dir() / "cache")

    def postprocess(self, model_outputs, multi_label=False):
        candidate_labels = [outputs["candidate_label"] for outputs in model_outputs]
        sequences = [outputs["sequence"] for outputs in model_outputs]
        logits = np.concatenate([output["logits"].numpy() for output in model_outputs])
        N = logits.shape[0]
        n = len(candidate_labels)
        num_sequences = N // n
        reshaped_outputs = logits.reshape((num_sequences, n, -1))

        if multi_label or len(candidate_labels) == 1:
            # softmax over the entailment vs. contradiction dim for each label independently
            entailment_id = self.entailment_id
            contradiction_id = -1 if entailment_id == 0 else 0
            entail_contr_logits = reshaped_outputs[..., [contradiction_id, entailment_id]]
            scores = np.exp(entail_contr_logits) / np.exp(entail_contr_logits).sum(-1, keepdims=True)
            scores = scores[..., 1]
        else:
            # softmax the "entailment" logits over all candidate labels
            entail_logits = reshaped_outputs[..., self.entailment_id]
            scores = np.exp(entail_logits) / np.exp(entail_logits).sum(-1, keepdims=True)

        top_inds = list(reversed(scores[0].argsort()))
        return {
            "sequence": sequences[0],
            "labels": [candidate_labels[i] for i in top_inds],
            "scores": scores[0, top_inds].tolist(),
            "entail_logits": reshaped_outputs[..., self.entailment_id][0, top_inds],
        }

    classifier.postprocess = MethodType(postprocess, classifier)
    return classifier


def parse(string, subs=None, pattern=r"\{([^}]+)\}"):
    subs = {k.upper(): v for k, v in subs.items()}

    def replace(match):
        key = match.group(1)
        return str(subs.get(key, match.group(0)))

    return re.sub(pattern, replace, string)


def xml(message: str, tag: str):
    return f"<{tag.upper()}>{message}</{tag.upper()}>"


def get_prompts(file_name: str):
    prompts_path = get_root_dir() / "prompts" / f"{file_name}.yaml"
    with open(prompts_path, "r") as file:
        prompts = yaml.safe_load(file)
    return prompts


def get_persona(id: str):
    persona_path = get_root_dir() / "patients" / id / ".yaml"
    with open(persona_path, "r") as file:
        persona = yaml.safe_load(file)
    return persona


def save_persona(persona: dict):
    persona_path = get_root_dir() / "patients" / f"{persona['id']}.yaml"
    with open(persona_path, "w") as file:
        file.write(yaml.safe_dump(persona, sort_keys=False))


def get_personas():
    personas_path = get_root_dir() / "patients"
    return [persona.split(".yaml")[0] for persona in os.listdir(personas_path)]


def hlin(start_hex, stop_hex, num):
    start = int(start_hex, 16)
    stop = int(stop_hex, 16)
    rang = np.linspace(start, stop, num)
    rang = [hex(int(r))[2:].rjust(2, "0") for r in rang]
    return rang


def hlin3(start, stop, num):
    tuples = list(zip(*[hlin(start[i : i + 2], stop[i : i + 2], num) for i in range(0, len(start), 2)]))
    hexes = ["".join(r) for r in tuples]
    return hexes


def make_system_prompt(persona: dict, prompts: dict, messages, personality, summary, activity=None):
    qa = "\n".join([f'Q: {m["type"]}; A: {m["content"]}' for m in persona["memories"]])
    system_prompt = parse(
        prompts["make_intention"],
        subs={
            "summary": summary,
            "personality": personality,
            "qa": qa,
            "quality": "GOOD",
            "specific": (
                prompts["intention_setting_activities"]["biographer_questions"][activity]["examples"]
                if activity is not None
                else "Sorry, no custom examples on this question. Just come up with your best, specific answer."
            ),
        },
    )

    return [{"role": "system", "content": system_prompt}] + messages[1:]


def get_importance(input: str, classifier, noise_mean=0, noise_sd=0):
    importance_strings = [
        "psychologically clinically important",
        "psychologically clinically critical",
        "psychologically clinically deep",
        "personally important",
        "personally critical",
        "personally deep",
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inference = classifier([input], importance_strings)
        noise = np.random.normal(noise_mean, noise_sd, 1)
    return [expit(np.mean(i["entail_logits"]) + noise) for i in inference][0][0]


def biographer(maker, importance_likert, prompts):
    contents = [f'{prompts["ask_question"]}\n\n{prompts["question_importance"][importance_likert]}'] + [
        message["content"] for message in maker.messages[1:]
    ]
    roles = ["system"] + [
        "user" if message["role"] == "assistant" else "assistant" for message in maker.messages[1:]
    ]
    return [{"role": role, "content": content} for role, content in zip(roles, contents)]


def combine_topic_memories(messages, topics, valences, importances):
    valence_beliefs = expit(logit((np.array(valences) + 1) / 2) + np.random.normal(0, 1, len(valences))) * 2 - 1
    importance_beliefs = np.array([np.random.uniform(0, i, 1)[0] for i in importances])
    memories = [
        {
            "type": "topic",
            "embed": topics[i].lower(),
            "content": messages[2 * i + 2]["content"],
            "metadata": {
                "valence": float(np.round(valences[i], 3)),
                "valence_belief": float(np.round(valence_beliefs[i], 3)),
                "importance": float(np.round(importances[i], 3)),
                "importance_belief": float(np.round(importance_beliefs[i], 3)),
            },
        }
        for i in range(len(topics))
    ]

    return memories


def get_skin_colors(top=("fce5b8", "f7d088"), bottom=("b3a789", "0d0800"), num=20):
    tops = hlin3(*top, num)
    bottoms = hlin3(*bottom, num)
    grid = list(zip(*[hlin3(t, b, num) for t, b in zip(tops, bottoms)]))
    skin_colors = ["#" + item for sublist in grid for item in sublist]
    return skin_colors
