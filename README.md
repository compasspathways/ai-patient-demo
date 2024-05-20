# Building virtual patients for training mental health professionals

This repository is the official implementation of "Building virtual patients for training mental health professionals"

## Talk to a Patient in 3 Steps

1. #### Install the requirements
    ```bash
    python -m venv venv-demo
    source venv-demo/bin/activate
    pip install -r requirements.txt
    ```

2. #### Configure the environment variables
    - `cp .env_sample .env`
    - Fill up the required variables (e.g. your openai api key) in the `.env` file
3. #### Run the app
    ```bash
    python chat_with.py leilani
    ```

## Make Your Own Patient in One Step
```bash
python make_patient.py
```

## Results

## Contributing
Any contribution to the code that improves the quality of chat with the patients or improves the performance is highly appreciated 🙏