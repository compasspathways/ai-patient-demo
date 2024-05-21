import openai
import tiktoken

from . import patient_maker_utils


class PatientMaker:
    def __init__(self):
        self.client = openai.OpenAI()
        self.messages = []
        self.model = "gpt-4-1106-preview"

    def command(self, string: str) -> str:
        system_message = [{"role": "system", "content": string}]
        openai_response = self.client.chat.completions.create(model=self.model, messages=system_message)
        return openai_response.choices[0].message.content

    def hear(self):
        openai_response = self.client.chat.completions.create(model=self.model, messages=self.messages)
        message = openai_response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": message})

    def say(self, message: str):
        self.messages.append({"role": "user", "content": message})

    def talkturn(self, message: str):
        self.say(message)
        self.hear()

    def print_messages(self):
        for message in self.messages:
            if message["role"] == "assistant":
                print(message["content"])

    @staticmethod
    def write(form: str, name: str):
        with open(patient_maker_utils.get_root_dir() / "forms" / f"{name}.txt", "w") as file:
            file.write(form)

    @staticmethod
    def count(input: str) -> int:
        return len(tiktoken.get_encoding("cl100k_base").encode(input))
