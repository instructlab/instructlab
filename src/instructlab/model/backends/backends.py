# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Optional
import abc
import logging
import pathlib

LLAMA_CPP = "llama-cpp"
SUPPORTED_BACKENDS = frozenset({LLAMA_CPP})


class BackendServer(abc.ABC):
    """Base class for a serving backend"""

    def __init__(
        self,
        logger: logging.Logger,
        model_path: pathlib.Path,
        host: str,
        port: int,
        **kwargs,
    ) -> None:
        self.logger = logger
        self.model_path = model_path
        self.host = host
        self.port = port

    @abc.abstractmethod
    def run(self):
        """Run serving backend"""

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
    # Standard
    import struct

    # Third Party
    from gguf.constants import GGUF_MAGIC

    with open(model_path, "rb") as f:
        first_four_bytes = f.read(4)

    # Unpack first four bytes (this is where the GGUF's magic number is) as little endian uint32
    first_four_bytes_int = struct.unpack("<I", first_four_bytes)[0]

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
    try:
        is_gguf = is_model_gguf(model_path)
    except Exception as e:
        raise ValueError(f"Failed to determine whether the model is a GGUF format: {e}")

    if is_gguf:
        return LLAMA_CPP
    else:
        raise ValueError(
            f"The model file {model_path} is not a GGUF format. Unsupported."
        )


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
    if backend == "":
        logger.debug(
            f"Backend is not set using auto-detected value: {auto_detected_backend}"
        )
        backend = auto_detected_backend
    # If the backend was set using the --backend flag, validate it.
    else:
        logger.debug(f"Validating '{backend}' backend")
        validate_backend(backend)
        # If the backend was set explicitly, but we detected the model should use a different backend, raise an error
        if backend != auto_detected_backend:
            raise ValueError(
                f"The backend '{backend}' was set explicitly, but the model was detected as '{auto_detected_backend}'."
            )

    return backend
