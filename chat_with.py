import pathlib
import sys

import gradio as gr
import yaml
from gradio.components import Chatbot

from source.patient import Patient

patient_id = sys.argv[1]

try:
    with open(pathlib.Path("patients") / f"{patient_id}.yaml", "r") as file:
        patient_persona = yaml.safe_load(file)
except FileNotFoundError:
    print(
        f">>> Given patient id: {patient_id} is not found in the patients folder! Please either create it using "
        " patient maker or use on of the existing ones!"
    )
    raise


with open(pathlib.Path("prompts/chat.yaml"), "r") as file:
    prompts = yaml.safe_load(file)

initial_conversation = prompts["initial_conversation"]
initial_conversation[0]["content"] = initial_conversation[0]["content"].replace(
    "[PATIENT_NAME]", patient_persona["name"]
)

patient = Patient(patient_persona, prompts, initial_conversation)


def patient_response(message, history):
    response, messages, metadata = patient.response(message)
    return response


app = gr.ChatInterface(
    patient_response,
    chatbot=Chatbot(
        label="Chatbot",
        scale=1,
        height=200,
        value=[
            [therapist["content"], patient["content"]]
            for therapist, patient in zip(initial_conversation[0::2], initial_conversation[1::2])
        ],
    ),
    title=f"AI Patient: {patient_persona['name']} 👤",
    description=f"{patient_persona['description']}",
    submit_btn="Send",
    retry_btn=None,
    undo_btn=None,
)

app.launch()
