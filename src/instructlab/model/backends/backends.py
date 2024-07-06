# SPDX-License-Identifier: Apache-2.0

# Standard
from time import sleep
from types import FrameType
from typing import Optional, Tuple
import abc
import contextlib
import logging
import mmap
import multiprocessing
import pathlib
import signal
import socket
import struct
import subprocess
import sys
import typing

# Third Party
from uvicorn import Config
import click
import fastapi
import httpx
import uvicorn

# Local
from ...client import ClientException, check_api_base, list_models
from ...configuration import _serve as serve_config
from ...configuration import get_api_base, get_model_family
from ...utils import split_hostport

logger = logging.getLogger(__name__)

LLAMA_CPP = "llama-cpp"
VLLM = "vllm"
SUPPORTED_BACKENDS = frozenset({LLAMA_CPP, VLLM})
API_ROOT_WELCOME_MESSAGE = "Hello from InstructLab! Visit us at https://instructlab.ai"
CHAT_TEMPLATE_AUTO = "auto"
CHAT_TEMPLATE_TOKENIZER = "tokenizer"
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


class Closeable(typing.Protocol):
    def close(self) -> None: ...


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
        model_family: str,
        model_path: pathlib.Path,
        chat_template: str,
        api_base: str,
        host: str,
        port: int,
    ) -> None:
        self.model_family = model_family
        self.model_path = model_path
        self.chat_template = chat_template if chat_template else CHAT_TEMPLATE_AUTO
        self.api_base = api_base
        self.host = host
        self.port = port
        self.resources: list[Closeable] = []

    @abc.abstractmethod
    def run(self):
        """Run serving backend in foreground (ilab model serve)"""

    @abc.abstractmethod
    def run_detached(
        self, http_client: httpx.Client | None = None, background: bool = True
    ) -> str:
        """Run serving backend in background ('ilab model chat' when server is not running)"""

    def shutdown(self):
        """Shutdown serving backend"""

        safe_close_all(self.resources)

    def register_resources(self, resources: typing.Iterable[Closeable]) -> None:
        self.resources.extend(resources)


def safe_close_all(resources: typing.Iterable[Closeable]):
    for resource in resources:
        with contextlib.suppress(Exception):
            resource.close()


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


def determine_backend(model_path: pathlib.Path) -> Tuple[str, str]:
    """
    Determine the backend to use based on the model file properties.
    Args:
        model_path (Path): The path to the model file.
    Returns:
        Tuple[str, str]: A tuple containing two strings:
                        - The backend to use.
                        - The reason why the backend was selected.
    """

    if model_path.is_dir():
        if sys.platform != "linux":
            raise ValueError(
                "Model is a directory containing huggingface safetensors files but the system is not Linux. "
                "Using a directory with safetensors file will activate the vLLM serving backend, vLLM is only supported on Linux. "
                "If you want to run the model on a different system (e.g. macOS), please use a GGUF file."
            )
        # If the model is a directory, it's a VLLM model - it's kinda weak, but it's a start
        if sys.platform == "linux":
            logger.debug(
                f"Model is a directory and system is Linux, using {VLLM} backend."
            )
            return (
                VLLM,
                "model path is a directory containing huggingface safetensors files and running on Linux.",
            )

    # Check if the model is a GGUF file
    try:
        is_gguf = is_model_gguf(model_path)
    except Exception as e:
        raise ValueError(
            f"Failed to determine whether the model is a GGUF format: {e}"
        ) from e

    if is_gguf:
        logger.debug(f"Model is a GGUF file, using {LLAMA_CPP} backend.")
        return LLAMA_CPP, "model is a GGUF file."

    raise ValueError(
        f"The model file {model_path} is not a GGUF format nor a directory containing huggingface safetensors files. Cannot determine which backend to use. \n"
        f"Please use a GGUF file for {LLAMA_CPP} or a directory containing huggingface safetensors files for {VLLM}. \n"
        "Note that vLLM is only supported on Linux."
    )


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
    try:
        auto_detected_backend, auto_detected_backend_reason = determine_backend(
            model_path
        )
    except ValueError as e:
        raise ValueError(f"Cannot determine which backend to use: {e}") from e

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
        # If the backend was set explicitly, but we detected the model should use a different backend, raise an error
        if backend != auto_detected_backend:
            logger.warning(
                f"The serving backend '{backend}' was configured explicitly, but the provided model is not compatible with it. "
                f"The model was detected as '{auto_detected_backend}, reason: {auto_detected_backend_reason}'.\n"
                "The backend startup sequence will continue with the configured backend but might fail."
            )

    return backend


# TODO: This is only used by vLLM but should move to vllm.py
def shutdown_process(process: subprocess.Popen, timeout: int) -> None:
    """
    Shuts down a process

    Sends SIGINT and then after a timeout if the process still is not terminated sends a SIGKILL

    Args:
        process (subprocess.Popen): process of the vllm server

    Returns:
        Nothing
    """
    # vLLM responds to SIGINT by shutting down gracefully and reaping the children
    process.send_signal(signal.SIGINT)
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
    background=True,
    server_process_func=None,
) -> Tuple[
    Optional[multiprocessing.Process], Optional[subprocess.Popen], Optional[str]
]:
    """Checks if server is running, if not starts one as a subprocess. Returns the server process
    and the URL where it's available."""

    # pylint: disable=no-else-return
    if check_api_base(api_base, http_client):
        return (None, None, api_base)
    else:
        port = free_tcp_ipv4_port(host)
        logger.debug(f"Using available port {port} for temporary model serving.")

        host_port = f"{host}:{port}"
        temp_api_base = get_api_base(host_port)
        llama_cpp_server_process = None
        vllm_server_process = None

        if backend == VLLM:
            # TODO: resolve how the hostname is getting passed around the class and this function
            vllm_server_process = server_process_func(port, background)
            logger.info(f"Starting a temporary vLLM server at {temp_api_base}")
            count = 0
            # TODO should this be configurable?
            vllm_startup_timeout = 60
            while count < vllm_startup_timeout:
                try:
                    list_models(
                        api_base=temp_api_base,
                        http_client=http_client,
                    )
                    logger.info(f"vLLM engine successfully started at {temp_api_base}")
                    break
                except ClientException:
                    count += 1
                    logger.info(
                        f"Waiting for the vLLM server to start at {temp_api_base}, this might take a moment... Retries: {count}/{vllm_startup_timeout}"
                    )
                    sleep(5)

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
            logger.debug(f"Starting a temporary llama.cpp server at {temp_api_base}")

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


def get_model_template(
    model_family: str, model_path: pathlib.Path
) -> Tuple[str, str, str]:
    eos_token = "<|endoftext|>"
    bos_token = ""
    template = ""
    resolved_family = get_model_family(model_family, model_path)
    for template_dict in templates:
        if template_dict["model"] == resolved_family:
            template = template_dict["template"]
            if template_dict["model"] == "mixtral":
                eos_token = "</s>"
                bos_token = "<s>"

    return template, eos_token, bos_token


def get_uvicorn_config(app: fastapi.FastAPI, host: str, port: int) -> Config:
    return Config(
        app,
        host=host,
        port=port,
        log_level=logging.ERROR,
        limit_concurrency=2,  # Make sure we only serve a single client at a time
        timeout_keep_alive=0,  # prevent clients holding connections open (we only have 1)
    )


def select_backend(
    cfg: serve_config,
    backend: Optional[str] = None,
    model_path: pathlib.Path | None = None,
) -> BackendServer:
    # Local
    from .llama_cpp import Server as llama_cpp_server
    from .vllm import Server as vllm_server

    model_path = pathlib.Path(model_path or cfg.model_path)
    backend_name = backend if backend is not None else cfg.backend
    try:
        backend = get(model_path, backend_name)
    except ValueError as e:
        click.secho(f"Failed to determine backend: {e}", fg="red")
        raise click.exceptions.Exit(1)

    host, port = split_hostport(cfg.host_port)
    chat_template = cfg.chat_template
    if not chat_template:
        chat_template = CHAT_TEMPLATE_AUTO

    if backend == LLAMA_CPP:
        # Instantiate the llama server
        return llama_cpp_server(
            api_base=cfg.api_base(),
            model_path=model_path,
            chat_template=chat_template,
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
            model_family=cfg.vllm.llm_family,
            model_path=model_path,
            chat_template=chat_template,
            vllm_args=cfg.vllm.vllm_args,
            host=host,
            port=port,
        )
    click.secho(f"Unknown backend: {backend}", fg="red")
    raise click.exceptions.Exit(1)


def verify_template_exists(path):
    if not path.exists():
        raise FileNotFoundError("Chat template file does not exist: {}".format(path))

    if not path.is_file():
        raise IsADirectoryError(
            "Chat templates paths must point to a file: {}".format(path)
        )
