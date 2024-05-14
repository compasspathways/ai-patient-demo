# AI Patient Demo

-   ## Talk to a Patient in 3 Steps

    - ### 1. Install
        ```bash
        python -m venv venv-demo
        source venv-demo/bin/activate
        pip install -r requirements.txt
        ```

    - ### 2. Configure
        - `cp .env_sample .env`
        - Fill up the required variables (e.g. your openai api key) in the `.env` file

    - ### 3. Use
        ```bash
        python chat_with.py roberta
        ```

- ## Make Your Own Patient in 1 step
    ```bash
    python make_patient.py
    ```
