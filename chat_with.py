import pathlib
import sys

import gradio as gr
import yaml

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


patient = Patient(patient_persona, [], [])


def patient_response(message, history):
    return patient.response(message)[0]


gr.ChatInterface(
    patient_response,
    title="AI Patient",
    description=f"{patient_persona['description']}",
    submit_btn="Send",
    retry_btn=None,
    undo_btn=None,
).launch()
