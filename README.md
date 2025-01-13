# Building virtual patients for training mental health professionals

This repository is the official implementation of "Building virtual patients for training mental health professionals"

## Talk to a Patient in 3 Steps

1. #### Install the requirements
    ```bash
    python -m venv venv-demo
    source venv-demo/bin/activate
    pip install -r requirements.txt
    ```
    **Note 1:** Make sure your python version is 3.11 <br>
    **Note 2:** The installation is successfully tested with:
    - Python 3.11.1 on MacBookPro 2023 M3 Max with 64 GB RAM (Sonoma 14.5)
    - Python 3.11.5 on MacBookPro 2019 with 16 GB RAM (Monterey 12.7.4)

2. #### Configure the environment variables
    - `cp env_sample .env`
    - Fill up the required variables (e.g. your openai api key) in the `.env` file
3. #### Run the app
    ```bash
    python chat_with.py leilani
    ```
    - Note: You will see a url printed on the console; open that link and chat with your patient!

## Make Your Own Patient in One Step
```bash
python make_patient.py
```
This will create a patient with a random name and realistic memories and back stories in `patients` folder. The intake form will be in `forms` folder.
You can edit the fields of the yaml file in the `patients` folder and talk to the newly created patient with `python chat_with.py [patient_id]`



## Contributing
Any contribution to the code that improves the quality of chat with the patients or improves the performance is highly appreciated 🙏