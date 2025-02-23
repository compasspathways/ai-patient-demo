# %% Imports
import logging
import os
import random
import string
import warnings

import numpy as np
import python_avatars as pa
import sentiment3d

from source import patient_maker, patient_maker_utils

logger = logging.getLogger("ai-patient")

# %% Init
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    maker = patient_maker.PatientMaker()
    prompts = patient_maker_utils.get_prompts()
    classifier = patient_maker_utils.get_classifier()
    logger.info("Instantiating sentiment model ..")
    s3d = sentiment3d.Sentiment3D()
    num_memories = int(os.getenv("NUMBER_OF_MEMORIES_FOR_NEW_PATIENT", 10))

# %% Intake Form
logger.info("Making Intake Form ...")
maker.messages = [{"role": "system", "content": prompts["patient_intake"]["fill_in"]}]
for section, content in prompts["patient_intake"]["sections"].items():
    print(section)
    maker.talkturn(content)
logger.info("done!")

maker.print_messages()

# %% Basic fields
intake_form = "\n".join([message["content"] for message in maker.messages if message["role"] == "assistant"])
patient_name = maker.command(f'{prompts["get_name"]}\n{intake_form}')
maker.write(intake_form, f"{patient_name}")
summary = maker.command(prompts["summarize"] + patient_maker_utils.xml(intake_form, "Intake Form"))
personality = maker.command(prompts["make_personality"] + patient_maker_utils.xml(summary, "bio"))
description = (
    f"You can talk to {patient_name} about anything, using the Method of Inquiry. "
    "Feel free to get to know them, build trust, and help them prepare for the trial."
)
logger.info(f"Patient Name: {patient_name}")

# %% Memories
messages = [
    {
        "role": "system",
        "content": prompts["make_fact"]
        + patient_maker_utils.xml(summary, "bio")
        + patient_maker_utils.xml(personality, "personality"),
    }
]

fact = ""
topics, importances, valences = [], [], []
for idx_memory in range(num_memories):
    logger.info(f"Creating content for memory {idx_memory+1}/{num_memories} ...")
    importance = 0 if idx_memory == 0 else patient_maker_utils.get_importance(fact, classifier, 1, 1)
    importance_likert = 0 if idx_memory == 0 else 1 + int(np.round(4 * importance))
    openai_response = maker.client.chat.completions.create(
        model=maker.model, messages=patient_maker_utils.biographer(maker, importance_likert, prompts)
    )

    messages.append({"role": "user", "content": openai_response.choices[0].message.content})

    messages[0] = {
        "role": "system",
        "content": prompts["guess_topic"]
        + patient_maker_utils.xml(summary, "bio")
        + patient_maker_utils.xml(personality, "personality"),
    }
    topic = (
        maker.client.chat.completions.create(model=maker.model, messages=[messages[0]] + messages[-3:])
        .choices[0]
        .message.content
    )
    topic = topic.translate(str.maketrans("", "", string.punctuation)).lower()
    topics.append(topic)

    messages[0] = {
        "role": "system",
        "content": prompts["make_fact"]
        + patient_maker_utils.xml(summary, "bio")
        + patient_maker_utils.xml(personality, "personality"),
    }
    fact = maker.client.chat.completions.create(model=maker.model, messages=messages).choices[0].message.content
    messages.append({"role": "assistant", "content": fact})

    importances.append(np.round(importance, 3))
    valences.append(np.round(s3d.get_utterance_sentiment(fact)["valence"], 3))
logger.info("Finished!")

memories = patient_maker_utils.combine_topic_memories(messages, topics, valences, importances)


# %% Avatar
logger.info("Creating avatar for the patient ...")
skin_colors = patient_maker_utils.get_skin_colors()
skin_color = random.choice(skin_colors)
random_color = "#" + "".join([hex(random.randint(0, 256))[2:].rjust(2, "0") for _ in range(3)])

avatar = (
    pa.Avatar(
        style=pa.AvatarStyle.CIRCLE,
        background_color=pa.BackgroundColor.WHITE,
        eyebrows=pa.EyebrowType.DEFAULT_NATURAL,
        eyes=pa.EyeType.DEFAULT,
        nose=pa.NoseType.DEFAULT,
        mouth=pa.MouthType.SERIOUS,
        facial_hair=pa.FacialHairType.NONE,
        skin_color=skin_color,
        accessory=pa.AccessoryType.NONE,
        clothing=pa.ClothingType.SHIRT_SCOOP_NECK,
        clothing_color=random_color,
    )
    .render()
    .replace('width="264px" height="280px"', 'width="100%" height="100%"')
)


# %% Saving
logger.info("Saving patient persona ...")
persona = {
    "id": patient_name.lower(),
    "name": patient_name,
    "model_id": maker.model,
    "avatar": avatar,
    "description": description,
    "definition": {
        "summary": summary,
        "personality": personality,
    },
    "memories": memories,
}

patient_maker_utils.save_persona(persona)
logger.info("Success!")
