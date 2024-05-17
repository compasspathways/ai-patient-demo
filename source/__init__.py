import logging.config
import os

import dotenv

dotenv.load_dotenv(override=True)


# to suppress huggingface/tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# config logging
green, reset = "\x1b[33;42m", "\x1b[0m"
default_formatter = logging.Formatter(
    f"{green}[%(levelname)s][%(asctime)s]{reset} >>> %(message)s", "%d/%m/%Y %H:%M:%S"
)

file_handler = logging.handlers.RotatingFileHandler(os.getenv("LOG_FILE_PATH", "/tmp/ai-patient.log"))
file_handler.setLevel(int(os.getenv("FILE_LOGGING_LEVEL", logging.DEBUG)))

console_handler = logging.StreamHandler()
console_handler.setLevel(int(os.getenv("CONSOLE_LOGGING_LEVEL", logging.INFO)))

file_handler.setFormatter(default_formatter)
console_handler.setFormatter(default_formatter)

root_logger = logging.getLogger("ai-patient")

root_logger.setLevel(logging.DEBUG)
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)
root_logger.propagate = False
