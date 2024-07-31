# Standard
from time import sleep, time
from types import FrameType
from typing import Optional, Tuple
import abc
import contextlib
import logging
import multiprocessing
import pathlib
import signal
import socket
import subprocess
import typing

# Third Party
from uvicorn import Config
import fastapi
import httpx
import uvicorn

# Local
from ...client import check_api_base
from ...configuration import get_api_base, get_model_family

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


class Closeable(typing.Protocol):
    def close(self) -> None: ...


def safe_close_all(resources: typing.Iterable[Closeable]):
    for resource in resources:
        with contextlib.suppress(Exception):
            resource.close()


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


def verify_template_exists(path):
    if not path.exists():
        raise FileNotFoundError("Chat template file does not exist: {}".format(path))

    if not path.is_file():
        raise IsADirectoryError(
            "Chat templates paths must point to a file: {}".format(path)
        )


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
    logger.debug(f"Sending SIGINT to vLLM server PID {process.pid}")
    process.send_signal(signal.SIGINT)
    try:
        logger.debug("Waiting for vLLM server to shut down gracefully")
        process.wait(timeout)
    except subprocess.TimeoutExpired:
        logger.debug(
            f"Sending SIGKILL to vLLM server since timeout ({timeout}s) expired"
        )
        process.kill()


def ensure_server(
    backend: str,
    api_base: str,
    http_client=None,
    host="localhost",
    port=8000,
    background=True,
    server_process_func=None,
) -> Tuple[
    Optional[multiprocessing.Process], Optional[subprocess.Popen], Optional[str]
]:
    """Checks if server is running, if not starts one as a subprocess. Returns the server process
    and the URL where it's available."""

    logger.info(f"Trying to connect to model server at {api_base}")
    if check_api_base(api_base, http_client):
        return (None, None, api_base)
    port = free_tcp_ipv4_port(host)
    logger.debug(f"Using available port {port} for temporary model serving.")

    host_port = f"{host}:{port}"
    temp_api_base = get_api_base(host_port)
    vllm_server_process = None

    if backend == VLLM:
        # TODO: resolve how the hostname is getting passed around the class and this function
        vllm_server_process = server_process_func(port, background)
        logger.info("Starting a temporary vLLM server at %s", temp_api_base)
        count = 0
        # TODO should this be configurable?

        vllm_startup_max_attempts = 80  # Each call to check_api_base takes >2s + 2s sleep = ~5 mins of wait time
        start_time_secs = time()
        while count < vllm_startup_max_attempts:
            count += 1
            logger.info(
                "Waiting for the vLLM server to start at %s, this might take a moment... Attempt: %s/%s",
                temp_api_base,
                count,
                vllm_startup_max_attempts,
            )
            if check_api_base(temp_api_base, http_client):
                logger.info("vLLM engine successfully started at %s", temp_api_base)
                break
            if count == vllm_startup_max_attempts:
                logger.info(
                    "Gave up waiting for vLLM server to start at %s after %s attempts",
                    temp_api_base,
                    vllm_startup_max_attempts,
                )
                duration = round(time() - start_time_secs, 1)
                shutdown_process(vllm_server_process, 20)
                # pylint: disable=raise-missing-from
                raise ServerException(f"vLLM failed to start up in {duration} seconds")
            sleep(2)
    return (None, vllm_server_process, temp_api_base)
