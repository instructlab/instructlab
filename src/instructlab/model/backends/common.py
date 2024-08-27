# Standard
from typing import Tuple
import logging
import pathlib

# Local
from ...configuration import get_model_family

logger = logging.getLogger(__name__)

API_ROOT_WELCOME_MESSAGE = "Hello from InstructLab! Visit us at https://instructlab.ai"
CHAT_TEMPLATE_AUTO = "auto"
CHAT_TEMPLATE_TOKENIZER = "tokenizer"
LLAMA_CPP = "llama-cpp"
VLLM = "vllm"
templates = [
    {
        "model": "merlinite",
        "template": "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n' + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}",
    },
    {
        "model": "mixtral",
        "template": "{{ bos_token }}\n{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '[INST] ' + message['content'] + ' [/INST]' }}\n{% elif message['role'] == 'assistant' %}\n{{ message['content'] + eos_token}}\n{% endif %}\n{% endfor %}",
    },
]


def get_model_template(
    model_family: str, model_path: pathlib.Path
) -> Tuple[str, str, str]:
    eos_token = "<|endoftext|>"
    bos_token = ""
    template = ""
    resolved_family = get_model_family(model_family, model_path)
    logger.debug(
        "Searching hard coded model templates for model family %s's template",
        resolved_family,
    )
    for template_dict in templates:
        if template_dict["model"] == resolved_family:
            template = template_dict["template"]
            if template_dict["model"] == "mixtral":
                eos_token = "</s>"
                bos_token = "<s>"

    return template, eos_token, bos_token


def verify_template_exists(path):
    if not path.exists():
        raise FileNotFoundError("Chat template file does not exist: {}".format(path))

    if not path.is_file():
        raise IsADirectoryError(
            "Chat templates paths must point to a file: {}".format(path)
        )
