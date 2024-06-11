# SPDX-License-Identifier: Apache-2.0

# Standard
from contextlib import redirect_stderr, redirect_stdout
from time import sleep
import logging
import multiprocessing
import os
import random
import signal
import socket

# Third Party
from llama_cpp import llama_chat_format
from llama_cpp.server.app import create_app
from llama_cpp.server.settings import Settings
from uvicorn import Config
import llama_cpp.server.app as llama_app
import uvicorn

# Local
from .client import ClientException, list_models
from .config import get_api_base, get_model_family

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


class ServerException(Exception):
    """An exception raised when serving the API."""


class Server(uvicorn.Server):
    """Override uvicorn.Server to handle SIGINT."""

    def handle_exit(self, sig, frame):
        # type: (int, Optional[FrameType]) -> None
        if not is_temp_server_running() or sig != signal.SIGINT:
            super().handle_exit(sig=sig, frame=frame)


def ensure_server(
    logger,
    serve_config,
    tls_insecure,
    tls_client_cert,
    tls_client_key,
    tls_client_passwd,
    model_family,
):
    """Checks if server is running, if not starts one as a subprocess. Returns the server process
    and the URL where it's available."""
    try:
        api_base = serve_config.api_base()
        logger.debug(f"Trying to connect to {api_base}...")
        # pylint: disable=duplicate-code
        list_models(
            api_base=api_base,
            tls_insecure=tls_insecure,
            tls_client_cert=tls_client_cert,
            tls_client_key=tls_client_key,
            tls_client_passwd=tls_client_passwd,
        )
        return (None, None, None)
        # pylint: enable=duplicate-code
    except ClientException:
        tried_ports = set()
        mpctx = multiprocessing.get_context(None)
        # use a queue to communicate between the main process and the server process
        queue = mpctx.Queue()
        port = random.randint(1024, 65535)
        host = serve_config.host_port.rsplit(":", 1)[0]
        logger.debug(f"Trying port {port}...")

        # extract address provided in the config
        while not can_bind_to_port(host, port):
            logger.debug(f"Port {port} is not available.")
            # add the port to the map so that we can avoid using the same one
            tried_ports.add(port)
            port = random.randint(1024, 65535)
            while True:
                # if all the ports have been tried, exit
                if len(tried_ports) == 65535 - 1024:
                    # pylint: disable=raise-missing-from
                    raise SystemExit(
                        "No available ports to start the temporary server."
                    )
                if port in tried_ports:
                    logger.debug(f"Port {port} has already been tried.")
                    port = random.randint(1024, 65535)
                else:
                    break
        logger.debug(f"Port {port} is available.")

        host_port = f"{host}:{port}"
        temp_api_base = get_api_base(host_port)
        logger.debug(
            f"Connection to {api_base} failed. Starting a temporary server at {temp_api_base}..."
        )
        # create a temporary, throw-away logger
        log = logging.getLogger(host_port)
        log.setLevel(logging.FATAL)
        log.debug("creating new server")
        server_process = mpctx.Process(
            target=server,
            kwargs={
                "logger": log,
                "model_path": serve_config.model_path,
                "gpu_layers": serve_config.gpu_layers,
                "max_ctx_size": serve_config.max_ctx_size,
                "model_family": model_family,
                "port": port,
                "host": host,
                "queue": queue,
            },
        )
        server_process.start()

        # in case the server takes some time to fail we wait a bit
        count = 0
        while server_process.is_alive():
            sleep(0.1)
            if count > 10:
                break
            count += 1

        # if the queue is not empty it means the server failed to start
        if not queue.empty():
            # pylint: disable=raise-missing-from
            raise queue.get()

        return (server_process, temp_api_base, queue)


def server(
    logger,
    model_path,
    gpu_layers,
    max_ctx_size,
    model_family,
    threads=None,
    host="localhost",
    port=8000,
    queue=None,
):
    """Start OpenAI-compatible server"""
    settings = Settings(
        host=host,
        port=port,
        model=model_path,
        n_ctx=max_ctx_size,
        n_gpu_layers=gpu_layers,
        verbose=logger.level == logging.DEBUG,
    )
    if threads is not None:
        settings.n_threads = threads
    try:
        app = create_app(settings=settings)

        @app.get("/")
        def read_root():
            return {
                "message": "Hello from InstructLab! Visit us at https://instructlab.ai"
            }
    except ValueError as exc:
        if queue:
            queue.put(exc)
            queue.close()
            queue.join_thread()
            return
        raise ServerException(f"failed creating the server application: {exc}") from exc

    template = ""
    eos_token = "<|endoftext|>"
    bos_token = ""
    for template_dict in templates:
        if template_dict["model"] == get_model_family(model_family, model_path):
            template = template_dict["template"]
            if template_dict["model"] == "mixtral":
                eos_token = "</s>"
                bos_token = "<s>"
    try:
        llama_app._llama_proxy._current_model.chat_handler = (
            llama_chat_format.Jinja2ChatFormatter(
                template=template,
                eos_token=eos_token,
                bos_token=bos_token,
            ).to_chat_handler()
        )
    # pylint: disable=broad-exception-caught
    except Exception as exc:
        if queue:
            queue.put(exc)
            queue.close()
            queue.join_thread()
            return
        raise ServerException(f"failed creating the server application: {exc}") from exc

    logger.info("Starting server process, press CTRL+C to shutdown server...")
    logger.info(
        f"After application startup complete see http://{host}:{port}/docs for API."
    )

    config = Config(
        app,
        host=host,
        port=port,
        log_level=logging.ERROR,
        limit_concurrency=2,  # Make sure we only serve a single client at a time
        timeout_keep_alive=0,  # prevent clients holding connections open (we only have 1)
    )
    s = Server(config)

    # If this is not the main process, this is the temp server process that ran in the background
    # after `ilab chat` was executed.
    # In this case, we want to redirect stdout to null to avoid cluttering the chat with messages
    # returned by the server.
    if is_temp_server_running():
        # TODO: redirect temp server logs to a file instead of hidding the logs completely
        # Redirect stdout and stderr to null
        with (
            open(os.devnull, "w", encoding="utf-8") as f,
            redirect_stdout(f),
            redirect_stderr(f),
        ):
            s.run()
    else:
        s.run()

    if queue:
        queue.close()
        queue.join_thread()


def can_bind_to_port(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except socket.error:
            return False


def is_temp_server_running():
    """Check if the temp server is running."""
    return multiprocessing.current_process().name != "MainProcess"
