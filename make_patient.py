# %%
import random
import string

import dotenv
import numpy as np
import python_avatars as pa
import sentiment3d

from source import patient_maker, utils

# %%
dotenv.load_dotenv(override=True)
maker = patient_maker.PatientMaker()
prompts = utils.get_prompts()
classifier = utils.get_classifier()
s3d = sentiment3d.Sentiment3D()

num_memories = 3

# %%
print(">>> Making Intake Form ...")
maker.messages = [{"role": "system", "content": prompts["patient_intake"]["fill_in"]}]
for section, content in prompts["patient_intake"]["sections"].items():
    print(section)
    maker.talkturn(content)
print(">>> done!")

maker.print_messages()

# %%
intake_form = "\n".join([message["content"] for message in maker.messages if message["role"] == "assistant"])
patient_name = maker.command(f'{prompts["get_name"]}\n{intake_form}')
maker.write(intake_form, f"{patient_name}")
print(patient_name)

# %%
summary = maker.command(prompts["summarize"] + utils.xml(intake_form, "Intake Form"))
personality = maker.command(prompts["make_personality"] + utils.xml(summary, "bio"))


# %%

topics, importances, valences = [], [], []

messages = [
    {
        "role": "system",
        "content": prompts["make_fact"] + utils.xml(summary, "bio") + utils.xml(personality, "personality"),
    }
]


for i in range(num_memories):
    print(f">>> Creating content for memory {i+1}/3 ...")
    importance = 0 if i == 0 else utils.get_importance(fact, classifier, 1, 1)
    importance_likert = 0 if i == 0 else 1 + int(np.round(4 * importance))
    openai_response = maker.client.chat.completions.create(
        model=maker.model, messages=utils.biographer(maker, importance_likert, prompts)
    )

    messages.append({"role": "user", "content": openai_response.choices[0].message.content})

    messages[0] = {
        "role": "system",
        "content": prompts["guess_topic"] + utils.xml(summary, "bio") + utils.xml(personality, "personality"),
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
        "content": prompts["make_fact"] + utils.xml(summary, "bio") + utils.xml(personality, "personality"),
    }
    fact = maker.client.chat.completions.create(model=maker.model, messages=messages).choices[0].message.content
    messages.append({"role": "assistant", "content": fact})

    importances.append(np.round(importance, 3))
    valences.append(np.round(s3d.get_utterance_sentiment(fact)["valence"], 3))
print(">>> Finished!")

i = 0
for m in messages[1:]:
    if m["role"] == "assistant":
        print("Answer:    " + m["content"] + "\n" + str(i))
    else:
        print("Question:  " + m["content"])
        print("Topic:     " + topics[i])
        print("Valence:   " + str(valences[i]) + ", Importance:  " + str(importances[i]))
        i += 1

memories = utils.combine_topic_memories(messages, topics, valences, importances)


# %%
print(">>> Creating avatar for the patient ...")
skin_colors = utils.get_skin_colors()
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


# %%
print(">>> Saving patient persona ...")
persona_id = patient_name.lower()
persona = {
    "id": persona_id,
    "name": patient_name,
    "model": maker.model,
    "summary": summary,
    "personality": personality,
    "avatar": avatar,
    "description": "<DEFINE ME>",
    "agent_type": "patient",
    "user_type": "therapist",
    "memories": memories,
}

utils.save_persona(persona)

# %%
activities = prompts["intention_setting_activities"]["biographer_questions"]
for activity in activities:
    for question in activities[activity]["questions"]:
        messages = utils.make_system_prompt(persona, prompts, messages, personality, summary, activity)
        maker.talkturn(question)

qa = "\n".join([("Q: " if m["role"] == "user" else "A: ") + m["content"] for m in messages[1:]])
system_prompt = utils.parse(
    prompts["intention_summary"], subs={"summary": summary, "personality": personality, "qa": qa}
)
intention_summary = maker.command(system_prompt)


# %%

persona["intention"] = {}
persona["intention_summary"] = intention_summary
q = [m["content"] for m in messages[1:]][::2]
a = [m["content"] for m in messages[1:]][1::2]
persona["search"] = {"topic": dict(zip(q, a))}
utils.save_persona(persona)
