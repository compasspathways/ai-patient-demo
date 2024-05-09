import pathlib
import sys

import gradio as gr
import yaml
from utils import Patient

patient_id = sys.argv[1]

with open(pathlib.Path(__file__).parent / "configs" / f"{patient_id}.yaml", "r") as file:
    patient_persona = yaml.safe_load(file)

patient = Patient(patient_id, patient_persona)


def patient_response(message, history):
    return patient.respond_to(message)


gr.ChatInterface(
    patient_response,
    title=f"AI Patient",
    description=f"Talking to {patient_id.title()} 👤 ",
    submit_btn="Send",
    retry_btn=None,
    undo_btn=None,
).launch()
