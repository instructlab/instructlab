# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Optional, Tuple
import logging
import pathlib
import sys

# Local
from ...configuration import _serve as serve_config
from ...utils import is_model_gguf, is_model_safetensors
from .common import CHAT_TEMPLATE_AUTO, LLAMA_CPP, VLLM
from .server import BackendServer

logger = logging.getLogger(__name__)

SUPPORTED_BACKENDS = frozenset({LLAMA_CPP, VLLM})


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


def check_model_path_exists(model_path: pathlib.Path) -> None:
    if not model_path.exists():
        error_message = f"{model_path} does not exist. Please download model first."
        print(f"\033[91m{error_message}\033[0m")
        raise FileNotFoundError(error_message)


def get_backend_from_values(
    host,
    port,
    model_path,
    backend_name,
    chat_template,
    api_base,
    gpu_layers,
    max_ctx_size,
    vllm_args,
    max_startup_attempts,
    model_family,
    vllm_model_family,
    log_file,
) -> BackendServer:
    # Local
    from .llama_cpp import Server as llama_cpp_server
    from .vllm import Server as vllm_server

    model_path = pathlib.Path(model_path)
    check_model_path_exists(model_path)
    try:
        backend = get(model_path, backend_name)
    except ValueError as e:
        print(f"\033[91mFailed to determine backend: {e}\033[0m")
        sys.exit(1)

    if not chat_template:
        chat_template = CHAT_TEMPLATE_AUTO

    if backend == LLAMA_CPP:
        # Instantiate the llama server
        return llama_cpp_server(
            api_base=api_base,
            model_path=model_path,
            chat_template=chat_template,
            gpu_layers=gpu_layers,
            max_ctx_size=max_ctx_size,
            model_family=model_family,
            host=host,
            port=port,
            log_file=log_file,
            num_threads=None,  # exists only as a flag not a config
        )
    if backend == VLLM:
        # Instantiate the vllm server
        return vllm_server(
            api_base=api_base,
            model_family=vllm_model_family,
            model_path=model_path,
            chat_template=chat_template,
            vllm_args=vllm_args,
            host=host,
            port=port,
            max_startup_attempts=max_startup_attempts,
            log_file=log_file,
        )
    print(f"\033[91mUnknown backend: {backend}\033[0m")
    sys.exit(1)


def select_backend(
    cfg: serve_config,
    backend: Optional[str] = None,
    model_path: pathlib.Path | None = None,
    log_file: pathlib.Path | None = None,
) -> BackendServer:
    logger.debug("Selecting backend for model %s", model_path)

    return get_backend_from_values(
        host=cfg.server.host,
        port=cfg.server.port,
        model_path=model_path if model_path is not None else cfg.model_path,
        backend_name=backend if backend is not None else cfg.backend,
        chat_template=cfg.chat_template,
        api_base=cfg.api_base(),
        gpu_layers=cfg.llama_cpp.gpu_layers,
        max_ctx_size=cfg.llama_cpp.max_ctx_size,
        vllm_args=cfg.vllm.vllm_args,
        max_startup_attempts=cfg.vllm.max_startup_attempts,
        vllm_model_family=cfg.vllm.llm_family,
        model_family=cfg.llama_cpp.llm_family,
        log_file=log_file,
    )
