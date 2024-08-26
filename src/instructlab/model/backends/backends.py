# SPDX-License-Identifier: Apache-2.0

# Standard
from time import monotonic, sleep, time
from types import FrameType
from typing import Optional, Tuple
import abc
import contextlib
import json
import logging
import multiprocessing
import os
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
from ...client import check_api_base
from ...configuration import _serve as serve_config
from ...configuration import get_api_base, get_model_family
from ...utils import split_hostport
from .common import CHAT_TEMPLATE_AUTO, LLAMA_CPP, VLLM, templates

logger = logging.getLogger(__name__)

SUPPORTED_BACKENDS = frozenset({LLAMA_CPP, VLLM})


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
        self,
        http_client: httpx.Client | None = None,
        background: bool = True,
        foreground_allowed: bool = False,
        max_startup_retries: int = 0,
    ) -> str:
        """Run serving backend in background ('ilab model chat' when server is not running)"""

    def shutdown(self):
        """Shutdown serving backend"""

        safe_close_all(self.resources)

    def register_resources(self, resources: typing.Iterable[Closeable]) -> None:
        self.resources.extend(resources)

    @abc.abstractmethod
    def get_backend_type(self):
        """Return which type of backend this is, llama-cpp or vllm"""


def safe_close_all(resources: typing.Iterable[Closeable]):
    for resource in resources:
        with contextlib.suppress(Exception):
            resource.close()


def is_model_safetensors(model_path: pathlib.Path) -> bool:
    """Check if model_path is a valid safe tensors directory

    Check if provided path to model represents directory containing a safetensors representation
    of a model. Directory must contain a specific set of files to qualify as a safetensors model directory
    Args:
        model_path (Path): The path to the model directory
    Returns:
        bool: True if the model is a safetensors model, False otherwise.
    """
    try:
        files = list(model_path.iterdir())
    except (FileNotFoundError, NotADirectoryError, PermissionError) as e:
        logger.debug("Failed to read directory: %s", e)
        return False

    # directory should contain either .safetensors or .bin files to be considered valid
    if not any(file.suffix in (".safetensors", ".bin") for file in files):
        logger.debug("'%s' has no *.safetensors or *.bin files", model_path)
        return False

    basenames = {file.name for file in files}
    requires_files = {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    diff = requires_files.difference(basenames)
    if diff:
        logger.debug("'%s' is missing %s", model_path, diff)
        return False

    for file in model_path.glob("*.json"):
        try:
            with file.open(encoding="utf-8") as f:
                json.load(f)
        except (PermissionError, json.JSONDecodeError) as e:
            logger.debug("'%s' is not a valid JSON file: e", file, e)
            return False

    # TODO: add check for safetensors file header (?)
    return True


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

    try:
        with model_path.open("rb") as f:
            first_four_bytes = f.read(4)

        # Convert the first four bytes to an integer
        first_four_bytes_int = int(struct.unpack("<I", first_four_bytes)[0])

        return first_four_bytes_int == GGUF_MAGIC
    except IsADirectoryError as exc:
        logger.debug(f"GGUF Path is a directory, returning {exc}")
        return False


def determine_backend(model_path: pathlib.Path) -> Tuple[str, str]:
    """
    Determine the backend to use based on the model file properties.
    Args:
        model_path (Path): The path to the model file/directory.
    Returns:
        Tuple[str, str]: A tuple containing two strings:
                        - The backend to use.
                        - The reason why the backend was selected.
    """
    if model_path.is_dir() and is_model_safetensors(model_path):
        if sys.platform == "linux":
            logger.debug(
                f"Model is huggingface safetensors and system is Linux, using {VLLM} backend."
            )
            return (
                VLLM,
                "model path is a directory containing huggingface safetensors files and running on Linux.",
            )
        raise ValueError(
            "Model is a directory containing huggingface safetensors files but the system is not Linux. "
            "Using a directory with safetensors file will activate the vLLM serving backend, vLLM is only supported on Linux. "
            "If you want to run the model on a different system (e.g. macOS), please use a GGUF file."
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


def get_max_stable_vram_wait(timeout: int) -> int:
    # Internal env variable for CI adjustment / disablement
    env_name = "ILAB_MAX_STABLE_VRAM_WAIT"
    wait_str = os.getenv(env_name)
    wait = timeout
    try:
        if wait_str:
            wait = int(wait_str)
    except ValueError:
        logger.warning(
            'Ignoring invalid timeout value for %s ("%s")', env_name, wait_str
        )

    return wait


# TODO: This is only used by vLLM but should move to vllm.py
def shutdown_process(process: subprocess.Popen, timeout: int) -> None:
    """
    Shuts down a process

    Sends SIGINT and then after a timeout if the process still is not terminated sends
    a SIGKILL. Finally, a SIGKILL is sent to the process group in case any child processes
    weren't cleaned up.

    Args:
        process (subprocess.Popen): process of the vllm server

    Returns:
        Nothing
    """
    # vLLM responds to SIGINT by shutting down gracefully and reaping the children
    logger.debug(f"Sending SIGINT to vLLM server PID {process.pid}")
    process_group_id = os.getpgid(process.pid)
    process.send_signal(signal.SIGINT)
    try:
        logger.debug("Waiting for vLLM server to shut down gracefully")
        process.wait(timeout)
    except subprocess.TimeoutExpired:
        logger.debug(
            f"Sending SIGKILL to vLLM server since timeout ({timeout}s) expired"
        )
        process.kill()

    # Attempt to cleanup any remaining child processes
    # Make sure process_group is legit (> 1) before trying
    if process_group_id and process_group_id > 1:
        try:
            os.killpg(process_group_id, signal.SIGKILL)
            logger.debug("Sent SIGKILL to vLLM process group")
        except ProcessLookupError:
            logger.debug("Nothing left to clean up with the vLLM process group")
    else:
        logger.debug("vLLM process group id not found")

    # Various facilities of InstructLab rely on multiple successive start/stop
    # cycles. Since vLLM relies on stable VRAM for estimating capacity, residual
    # reclamation activity can lead to crashes on start. To prevent this add a
    # short delay (typically ~ 10 seconds, max 30) to verify stability.
    #
    # Ideally a future enhancement would be contributed to vLLM to more gracefully
    # handle this condition.
    wait_for_stable_vram(get_max_stable_vram_wait(30))


def wait_for_stable_vram(timeout: int):
    logger.info("Waiting for GPU VRAM reclamation...")
    supported, stable = wait_for_stable_vram_cuda(timeout)
    if not supported:
        # TODO add support for intel
        sleep(timeout)
        return
    if not stable:
        # Only for debugging since recovery is likely after additional start delay
        logger.debug(
            "GPU VRAM did not stabilize during max timeout (%d seconds)", timeout
        )


def wait_for_stable_vram_cuda(timeout: int) -> Tuple[bool, bool]:
    if timeout == 0:
        logger.debug("GPU vram stability check disabled with 0 max wait value")
        return True, True

    # Third Party
    import torch

    # Fallback to a constant sleep if we don't have support for the device
    if not torch.cuda.is_available():
        return False, False
    start_time = monotonic()
    stable_samples = 0
    last_free = 0
    try:
        while True:
            free_memory = 0
            try:
                # TODO In the future this should be enhanced to better handle
                # GPU partitioning. However, to do so will require that serve
                # assign specific GPUs to vLLM, so that the same device pool is
                # analyzed.
                for i in range(torch.cuda.device_count()):
                    device = torch.device(f"cuda:{i}")
                    free_memory += torch.cuda.mem_get_info(device)[0]
            except Exception:  # pylint: disable=broad-exception-caught
                logger.warning(
                    "Could not determine CUDA memory, falling back to general sleep"
                )
                return False, False

            # Wait for free memory to stop growing indicating the release of
            # vram after vLLM shutdown is complete. Wait for 5 successful
            # samples where this is true, but ignore any spurious readings that
            # occur between those samples. In the future we may be able to
            # optimize this by checking a few strictly successive samples.
            if free_memory <= last_free:
                stable_samples += 1
                logger.debug(
                    "GPU free vram stable (stable count %d, free %d, last free %d)",
                    stable_samples,
                    free_memory,
                    last_free,
                )
                if stable_samples > 5:
                    logger.debug(
                        "Successful sample recorded, (stable count %d, free %d, last free %d)",
                        stable_samples,
                        free_memory,
                        last_free,
                    )
                    return True, True
            else:
                if last_free != 0:
                    logger.debug(
                        "GPU free vram still growing (free %d, last free %d)",
                        free_memory,
                        last_free,
                    )

            if monotonic() - start_time > timeout:
                return True, False

            last_free = free_memory
            sleep(1)
    finally:
        try:
            torch.cuda.empty_cache()
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.debug("Could not free cuda cache: %s", e)


def ensure_server(
    backend: str,
    api_base: str,
    http_client=None,
    host="localhost",
    port=8000,
    background=True,
    foreground_allowed=False,
    server_process_func=None,
    max_startup_attempts=None,
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
        # Each call to check_api_base takes >2s + 2s sleep
        # Default to 120 if not specified (~8 mins of wait time)
        vllm_startup_max_attempts = max_startup_attempts or 120
        start_time_secs = time()
        while count < vllm_startup_max_attempts:
            count += 1
            # Check if the process is still alive
            if vllm_server_process.poll():
                if foreground_allowed and background:
                    raise ServerException(
                        "vLLM failed to start.  Retry with --enable-serving-output to learn more about the failure."
                    )
                raise ServerException("vLLM failed to start.")
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
            max_startup_attempts=cfg.vllm.max_startup_attempts,
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


def start_backend(
    client,
    cfg: serve_config,
    backend: Optional[str] = None,
    model_path: pathlib.Path | None = None,
    background: bool = True,
    foreground_allowed: bool = False,
    max_startup_retries: int = 0,
) -> BackendServer:
    be = select_backend(cfg, backend=backend, model_path=model_path)
    try:
        be.run_detached(client, background, foreground_allowed, max_startup_retries)
    except Exception as exc:
        click.secho(f"Failed to start server: {exc}", fg="red")
        raise click.exceptions.Exit(1)
    return be
