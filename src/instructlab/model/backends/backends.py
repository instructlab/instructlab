# SPDX-License-Identifier: Apache-2.0

# Standard
from time import sleep
from types import FrameType
from typing import Optional, Tuple
import abc
import logging
import mmap
import multiprocessing
import pathlib
import signal
import socket
import struct
import subprocess
import sys

# Third Party
from uvicorn import Config
import click
import fastapi
import httpx
import uvicorn

# Local
from ...client import ClientException, list_models
from ...configuration import _serve as serve_config
from ...configuration import get_api_base
from ...utils import split_hostport

logger = logging.getLogger(__name__)

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

    def handle_exit(self, sig: int, frame: Optional[FrameType]) -> None:
        if not is_temp_server_running() or sig != signal.SIGINT:
            super().handle_exit(sig=sig, frame=frame)


class BackendServer(abc.ABC):
    """Base class for a serving backend"""

    def __init__(
        self,
        model_path: pathlib.Path,
        api_base: str,
        host: str,
        port: int,
    ) -> None:
        self.model_path = model_path
        self.api_base = api_base
        self.host = host
        self.port = port

    @abc.abstractmethod
    def run(self):
        """Run serving backend in foreground (ilab model serve)"""

    @abc.abstractmethod
    def run_detached(self, http_client: httpx.Client | None = None) -> str:
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
    # Third Party
    from gguf.constants import GGUF_MAGIC

    with open(model_path, "rb") as f:
        # Memory-map the file on the first 4 bytes (this is where the magic number is)
        mmapped_file = mmap.mmap(f.fileno(), length=4, access=mmap.ACCESS_READ)

        # Read the first 4 bytes
        first_four_bytes = mmapped_file.read(4)

        # Convert the first four bytes to an integer
        first_four_bytes_int = int(struct.unpack("<I", first_four_bytes)[0])

        # Close the memory-mapped file
        mmapped_file.close()

        return first_four_bytes_int == GGUF_MAGIC


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


def get(model_path: pathlib.Path, backend: str | None) -> str:
    """
    Get the backend to use based on the model file properties.
    Args:
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
    if backend is None:
        logger.debug(
            f"Backend is not set using auto-detected value: {auto_detected_backend}"
        )
        backend = auto_detected_backend
    # If the backend was set using the --backend flag, validate it.
    else:
        logger.debug(f"Validating '{backend}' backend")
        # TODO: keep this code logic and implement a `--force` flag to override the auto-detected backend
        # If the backend was set explicitly, but we detected the model should use a different backend, raise an error
        # if backend != auto_detected_backend:
        #     raise ValueError(
        #         f"The backend '{backend}' was set explicitly, but the model was detected as '{auto_detected_backend}'."
        #     )

    return backend


def shutdown_process(process: subprocess.Popen, timeout: int) -> None:
    """
    Shuts down a process

    Sends SIGTERM and then after a timeout if the process still is not terminated sends a SIGKILL

    Args:
        process (subprocess.Popen): process of the vllm server

    Returns:
        Nothing
    """
    process.terminate()
    try:
        process.wait(timeout)
    except subprocess.TimeoutExpired:
        process.kill()


def ensure_server(
    backend: str,
    api_base: str,
    http_client=None,
    host="localhost",
    port=8000,
    queue=None,
    server_process_func=None,
) -> Tuple[
    Optional[multiprocessing.Process], Optional[subprocess.Popen], Optional[str]
]:
    """Checks if server is running, if not starts one as a subprocess. Returns the server process
    and the URL where it's available."""

    try:
        logger.debug(f"Trying to connect to {api_base}...")
        # pylint: disable=duplicate-code
        list_models(
            api_base=api_base,
            http_client=http_client,
        )
        return (None, None, api_base)
        # pylint: enable=duplicate-code
    except ClientException:
        port = free_tcp_ipv4_port(host)
        logger.debug("Using %i", port)

       # host_port = f"{host}:{port}"
        temp_api_base = get_api_base(host, str(port))
        logger.debug(f"Starting a temporary server at {temp_api_base}...")
        llama_cpp_server_process = None
        vllm_server_process = None

        if backend == VLLM:
            # TODO: resolve how the hostname is getting passed around the class and this function
            vllm_server_process = server_process_func(port)
            count = 0
            # TODO should this be configurable?
            vllm_startup_timeout = 300
            while count < vllm_startup_timeout:
                sleep(1)
                try:
                    list_models(
                        api_base=temp_api_base,
                        http_client=http_client,
                    )
                    logger.debug(f"model at {temp_api_base} served on vLLM")
                    break
                except ClientException:
                    count += 1

            if count >= vllm_startup_timeout:
                shutdown_process(vllm_server_process, 20)
                # pylint: disable=raise-missing-from
                raise ServerException(
                    f"vLLM failed to start up in {vllm_startup_timeout} seconds"
                )

        elif backend == LLAMA_CPP:
            # server_process_func is a function! we invoke it here and pass the port that was determined
            # in this ensure_server() function
            llama_cpp_server_process = server_process_func(port)
            llama_cpp_server_process.start()

            # in case the server takes some time to fail we wait a bit
            logger.debug("Waiting for the server to start...")
            count = 0
            while llama_cpp_server_process.is_alive():
                sleep(0.1)
                try:
                    list_models(
                        api_base=temp_api_base,
                        http_client=http_client,
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

        return (llama_cpp_server_process, vllm_server_process, temp_api_base)


def free_tcp_ipv4_port(host: str) -> int:
    """Ask the OS for a random, ephemeral, and bindable TCP/IPv4 port

    Note: The idea of finding a free port is bad design and subject to
    race conditions. Instead vLLM and llama-cpp should accept port 0 and
    have an API to return the actual listening port. Or they should be able
    to use an existing socket like a systemd socket activation service.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[-1])


def is_temp_server_running():
    """Check if the temp server is running."""
    return multiprocessing.current_process().name != "MainProcess"


def get_uvicorn_config(app: fastapi.FastAPI, host: str, port: int) -> Config:
    return Config(
        app,
        host=host,
        port=port,
        log_level=logging.ERROR,
        limit_concurrency=2,  # Make sure we only serve a single client at a time
        timeout_keep_alive=0,  # prevent clients holding connections open (we only have 1)
    )


def select_backend(cfg: serve_config) -> BackendServer:
    # Local
    from .llama_cpp import Server as llama_cpp_server
    from .vllm import Server as vllm_server

    model_path = pathlib.Path(cfg.model_path)
    backend_name = cfg.backend
    try:
        backend = get(model_path, backend_name)
    except ValueError as e:
        click.secho(f"Failed to determine backend: {e}", fg="red")
        raise click.exceptions.Exit(1)

    host = cfg.host
    port = cfg.port

    if backend == LLAMA_CPP:
        # Instantiate the llama server
        return llama_cpp_server(
            api_base=cfg.api_base(),
            model_path=model_path,
            gpu_layers=cfg.llama_cpp.gpu_layers,
            max_ctx_size=cfg.llama_cpp.max_ctx_size,
            num_threads=None,  # exists only as a flag not a config
            model_family=cfg.llama_cpp.llm_family,
            host=host,
            port=port,
        )
    if backend == VLLM:
        # Instantiate the vllm server
        return vllm_server(
            api_base=cfg.api_base(),
            model_path=model_path,
            vllm_additional_args=cfg.vllm.vllm_additional_args,
            device=cfg.vllm.device,
            host=host,
            port=port,
            served_model_name=cfg.vllm.served_model_name,
            tensor_parallel_size=cfg.vllm.tensor_parallel_size,
            max_parallel_loading_workers=cfg.vllm.max_parallel_loading_workers,
            max_model_len=cfg.vllm.max_model_len,
        )
    click.secho(f"Unknown backend: {backend}", fg="red")
    raise click.exceptions.Exit(1)
