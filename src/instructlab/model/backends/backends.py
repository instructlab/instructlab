# SPDX-License-Identifier: Apache-2.0

# Standard
from time import sleep
from typing import Callable, Optional, Tuple
import abc
import logging
import mmap
import multiprocessing
import pathlib
import random
import signal
import socket
import struct
import sys

# Third Party
from uvicorn import Config
import uvicorn

# Local
from ...client import ClientException, list_models
from ...configuration import get_api_base
from ...configuration import _serve as serve_config
from ...utils import split_hostport

LLAMA_CPP = "llama-cpp"
VLLM = "vllm"
SUPPORTED_BACKENDS = frozenset({LLAMA_CPP, VLLM})
API_ROOT_WELCOME_MESSAGE = "Hello from InstructLab! Visit us at https://instructlab.ai"
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


class UvicornServer(uvicorn.Server):
    """Override uvicorn.Server to handle SIGINT."""

    def handle_exit(self, sig, frame):
        # type: (int, Optional[FrameType]) -> None
        if not is_temp_server_running() or sig != signal.SIGINT:
            super().handle_exit(sig=sig, frame=frame)


class BackendServer(abc.ABC):
    """Base class for a serving backend"""

    def __init__(
        self,
        logger: logging.Logger,
        model_path: pathlib.Path,
        api_base: str,
        host: str,
        port: int,
        **kwargs,
    ) -> None:
        self.logger = logger
        self.model_path = model_path
        self.api_base = api_base
        self.host = host
        self.port = port
        self.process = None

    @abc.abstractmethod
    def run(self):
        """Run serving backend in foreground (ilab model serve)"""

    @abc.abstractmethod
    def run_detached(
        self, tls_insecure, tls_client_cert, tls_client_key, tls_client_passwd
    ):
        """Run serving backend in background ('ilab model chat' when server is not running)"""

    @abc.abstractmethod
    def shutdown(self):
        """Shutdown serving backend"""


def is_model_gguf(model_path: pathlib.Path) -> bool:
    """
    Check if the file is a GGUF file.
    Args:
        model_path (Path): The path to the file.
    Returns:
        bool: True if the file is a GGUF file, False otherwise.
    """
    # pylint: disable=import-outside-toplevel
    # Third Party
    from gguf.constants import GGUF_MAGIC

    with open(model_path, "rb") as f:
        # Memory-map the file on the first 4 bytes (this is where the magic number is)
        mmapped_file = mmap.mmap(f.fileno(), length=4, access=mmap.ACCESS_READ)

        # Read the first 4 bytes
        first_four_bytes = mmapped_file.read(4)

        # Convert the first four bytes to an integer
        first_four_bytes_int = struct.unpack("<I", first_four_bytes)[0]

        # Close the memory-mapped file
        mmapped_file.close()

        return first_four_bytes_int == GGUF_MAGIC


def validate_backend(backend: str) -> None:
    """
    Validate the backend.
    Args:
        backend (str): The backend to validate.
    Raises:
        ValueError: If the backend is not supported.
    """
    # lowercase backend for comparison in case of user input like 'Llama'
    if backend.lower() not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Backend '{backend}' is not supported. Supported: {', '.join(SUPPORTED_BACKENDS)}"
        )


def determine_backend(model_path: pathlib.Path) -> str:
    """
    Determine the backend to use based on the model file properties.
    Args:
        model_path (Path): The path to the model file.
    Returns:
        str: The backend to use.
    """
    # Check if the model is a GGUF file

    # If the model is a directory, it's a VLLM model - it's kinda weak, but it's a start
    if model_path.is_dir() and sys.platform == "linux":
        return VLLM

    try:
        is_gguf = is_model_gguf(model_path)
    except Exception as e:
        raise ValueError(
            f"Failed to determine whether the model is a GGUF format: {e}"
        ) from e

    if is_gguf:
        return LLAMA_CPP

    raise ValueError(f"The model file {model_path} is not a GGUF format. Unsupported.")


def get(logger: logging.Logger, model_path: pathlib.Path, backend: str) -> str:
    """
    Get the backend to use based on the model file properties.
    Args:
        logger (Logger): The logger to use.
        model_path (Path): The path to the model file.
        backend (str): The backend that might have been pass to the CLI or set in config file.
    Returns:
        str: The backend to use.
    """
    # Check if the model is a GGUF file
    logger.debug(f"Auto-detecting backend for model {model_path}")
    auto_detected_backend = determine_backend(model_path)

    logger.debug(f"Auto-detected backend: {auto_detected_backend}")
    # When the backend is not set using the --backend flag, determine the backend automatically
    # 'backend' is optional so we still check for None or empty string in case 'config.yaml' hasn't
    # been updated via 'ilab config init'
    if backend is None or backend == "":
        logger.debug(
            f"Backend is not set using auto-detected value: {auto_detected_backend}"
        )
        backend = auto_detected_backend
    # If the backend was set using the --backend flag, validate it.
    else:
        logger.debug(f"Validating '{backend}' backend")
        validate_backend(backend)
        # TODO: keep this code logic and implement a `--force` flag to override the auto-detected backend
        # If the backend was set explicitly, but we detected the model should use a different backend, raise an error
        # if backend != auto_detected_backend:
        #     raise ValueError(
        #         f"The backend '{backend}' was set explicitly, but the model was detected as '{auto_detected_backend}'."
        #     )

    return backend


def ensure_server(
    logger: logging.Logger,
    server_process_func: Callable[[str], multiprocessing.Process],
    api_base: str,
    tls_insecure: bool,
    tls_client_cert: Optional[str] = None,
    tls_client_key: Optional[str] = None,
    tls_client_passwd: Optional[str] = None,
    host="localhost",
    port=8000,
    queue=None,
) -> Tuple[multiprocessing.Process, str]:
    """Checks if server is running, if not starts one as a subprocess. Returns the server process
    and the URL where it's available."""

    try:
        logger.debug(f"Trying to connect to {api_base}...")
        # pylint: disable=duplicate-code
        list_models(
            api_base=api_base,
            tls_insecure=tls_insecure,
            tls_client_cert=tls_client_cert,
            tls_client_key=tls_client_key,
            tls_client_passwd=tls_client_passwd,
        )
        return (None, None)
        # pylint: enable=duplicate-code
    except ClientException:
        tried_ports = set()
        port = random.randint(1024, 65535)
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
        logger.debug(f"Starting a temporary server at {temp_api_base}...")

        # server_process_func is a function! we invoke it here and pass the port that was determined
        # in this ensure_server() function
        server_process = server_process_func(port=port)
        server_process.start()

        # in case the server takes some time to fail we wait a bit
        logger.debug("Waiting for the server to start...")
        count = 0
        while server_process.is_alive():
            sleep(0.1)
            try:
                list_models(
                    api_base=temp_api_base,
                    tls_insecure=tls_insecure,
                    tls_client_cert=tls_client_cert,
                    tls_client_key=tls_client_key,
                    tls_client_passwd=tls_client_passwd,
                )
                break
            except ClientException:
                pass
            if count > 50:
                logger.error("failed to reach the API server")
                break
            count += 1

        logger.debug("Server started.")

        # if the queue is not empty it means the server failed to start
        if queue is not None and not queue.empty():
            # pylint: disable=raise-missing-from
            raise queue.get()

        return (server_process, temp_api_base)


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


def get_uvicorn_config(app: uvicorn.Server, host: str, port: int) -> Config:
    return Config(
        app,
        host=host,
        port=port,
        log_level=logging.ERROR,
        limit_concurrency=2,  # Make sure we only serve a single client at a time
        timeout_keep_alive=0,  # prevent clients holding connections open (we only have 1)
    )

def select_backend(logger: logging.Logger, serve_config: serve_config) -> BackendServer:
    from .llama_cpp import Server as llama_cpp_server
    from .vllm import Server as vllm_server
    backend_instance = None
    model_path = pathlib.Path(serve_config.model_path)
    backend_name = serve_config.backend
    try:
        backend = get(logger, model_path, backend_name)
    except ValueError as e:
        click.secho(f"Failed to determine backend: {e}", fg="red")
        raise click.exceptions.Exit(1)

    host, port = split_hostport(serve_config.host_port)

    if backend == LLAMA_CPP:
        # Instantiate the llama server
        backend_instance = llama_cpp_server(
            logger=logger,
            api_base=serve_config.api_base(),
            model_path=model_path,
            gpu_layers=serve_config.llama_cpp.gpu_layers,
            max_ctx_size=serve_config.llama_cpp.max_ctx_size,
            num_threads=None,  # exists only as a flag not a config
            model_family=serve_config.model_family,
            host=host,
            port=port,
        )

    if backend == VLLM:
        # Instantiate the vllm server
        backend_instance = vllm_server(
            logger=logger,
            api_base=serve_config.api_base(),
            model_path=model_path,
            model_family=serve_config.model_family,
            host=host,
            port=port,
        )
    return backend_instance
