# Standard
import logging
import multiprocessing
import random

# Third Party
from llama_cpp import llama_chat_format
from llama_cpp.server.app import create_app
from llama_cpp.server.settings import Settings
import llama_cpp.server.app as llama_app
import uvicorn

# Local
from .client import ClientException, list_models
from .config import get_api_base


class ServerException(Exception):
    """An exception raised when serving the API."""


def ensure_server(logger, serve_config):
    """Checks if server is running, if not starts one as a subprocess. Returns the server process and the URL where it's available."""
    try:
        api_base = serve_config.api_base()
        logger.debug(f"Trying to connect to {api_base}...")
        list_models(api_base)
        return (None, None)
    except ClientException:
        port = random.randint(1024, 65535)
        host_port = f"locahost:{port}"
        temp_api_base = get_api_base(host_port)
        logger.debug(
            f"Connection to {api_base} failed. Starting a temporary server at {temp_api_base}..."
        )
        # create a temporary, throw-away logger
        logger = logging.getLogger(host_port)
        logger.setLevel(logging.FATAL)
        server_process = multiprocessing.Process(
            target=server,
            kwargs={
                "logger": logger,
                "model_path": serve_config.model_path,
                "gpu_layers": serve_config.gpu_layers,
                "port": port,
            },
            daemon=True,
        )
        server_process.start()
        return (server_process, temp_api_base)


def server(logger, model_path, gpu_layers, host="localhost", port=8000):
    """Start OpenAI-compatible server"""
    settings = Settings(
        host=host,
        port=port,
        model=model_path,
        n_ctx=4096,
        n_gpu_layers=gpu_layers,
        verbose=logger.level == logging.DEBUG,
    )
    try:
        app = create_app(settings=settings)
    except ValueError as exc:
        raise ServerException(f"failed creating the server application: {exc}") from exc

    try:
        llama_app._llama_proxy._current_model.chat_handler = llama_chat_format.Jinja2ChatFormatter(
            template="{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n' + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}",
            eos_token="<|endoftext|>",
            bos_token="",
        ).to_chat_handler()
    # pylint: disable=broad-exception-caught
    except Exception as exc:
        raise ServerException(f"failed creating the server application: {exc}") from exc

    logger.info("Starting server process, press CTRL+C to shutdown server...")
    logger.info(
        f"After application startup complete see http://{host}:{port}/docs for API."
    )
    uvicorn.run(app, host=host, port=port, log_level=logging.ERROR)
