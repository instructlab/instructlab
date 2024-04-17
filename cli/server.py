# Standard
from multiprocessing import Queue
from time import sleep
import logging
import multiprocessing
import random
import socket
from cli import config

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


def ensure_server(
    logger,
    serve_config,
    tls_insecure,
    tls_client_cert,
    tls_client_key,
    tls_client_passwd,
):
    """Checks if server is running, if not starts one as a subprocess. Returns the server process
    and the URL where it's available."""
    try:
        api_base = serve_config.api_base()
        logger.debug(f"Trying to connect to {api_base}...")
        list_models(
            api_base=api_base,
            tls_insecure=tls_insecure,
            tls_client_cert=tls_client_cert,
            tls_client_key=tls_client_key,
            tls_client_passwd=tls_client_passwd,
        )
        return (None, None)
    except ClientException:
        tried_ports = set()
        # use a queue to communicate between the main process and the server process
        queue = Queue()
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
        logger = logging.getLogger(host_port)
        logger.setLevel(logging.FATAL)
        server_process = multiprocessing.Process(
            target=server,
            kwargs={
                "logger": logger,
                "model_path": serve_config.model_path,
                "gpu_layers": serve_config.gpu_layers,
                "max_ctx_size": serve_config.max_ctx_size,
                "port": port,
                "host": host,
                "queue": queue,
            },
            daemon=True,
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

        return (server_process, temp_api_base)


def server(
    logger,
    model_path,
    gpu_layers,
    max_ctx_size,
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
    current = config.read_config()
    if current.serve.model_path != model_path:
        # update cfg to have the newly set model_path
        current.serve.model_path = model_path
        config.write_config(current)
    if threads is not None:
        settings.n_threads = threads
    try:
        app = create_app(settings=settings)
    except ValueError as exc:
        if queue:
            queue.put(exc)
            return
        raise ServerException(f"failed creating the server application: {exc}") from exc

    try:
        llama_app._llama_proxy._current_model.chat_handler = llama_chat_format.Jinja2ChatFormatter(
            template="{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n' + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}",
            eos_token="<|endoftext|>",
            bos_token="",
        ).to_chat_handler()
    # pylint: disable=broad-exception-caught
    except Exception as exc:
        if queue:
            queue.put(exc)
            return
        raise ServerException(f"failed creating the server application: {exc}") from exc

    logger.info("Starting server process, press CTRL+C to shutdown server...")
    logger.info(
        f"After application startup complete see http://{host}:{port}/docs for API."
    )
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=logging.ERROR,
        limit_concurrency=2,  # Make sure we only serve a single client at a time
        timeout_keep_alive=0,  # prevent clients holding connections open (we only have 1)
    )


def can_bind_to_port(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except socket.error:
            return False
