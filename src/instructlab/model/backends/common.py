# Standard
import logging

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
