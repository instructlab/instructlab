# Standard
from pathlib import Path
import logging
import shutil

# First Party
from instructlab.configuration import DEFAULTS
from instructlab.model.backends.backends import is_model_gguf, is_model_safetensors

logger = logging.getLogger(__name__)


class RemovalError(Exception):
    """Custom exception for removal errors."""


def remove_model(model: str, model_dir: str):
    """Remove model and optionally clear the cache."""
    models_dir = Path(model_dir)
    model_path = models_dir / model
    base_dir_str = models_dir.name + "/"
    cache_model_path = model

    # Without <username> model dir and start with models/
    if model.startswith(base_dir_str):
        remove_models_path = model[len(base_dir_str) :]
        model_path = models_dir / remove_models_path
        cache_model_path = remove_models_path

    if not model_path.exists():
        raise RemovalError(f"Model {model} does not exist in {models_dir}.")

    if model_path.is_file() and is_model_gguf(model_path):
        pass
    elif model_path.is_dir() and is_model_safetensors(model_path):
        # With username model dir e.g. <username>/model
        if base_dir_str not in model:
            username_part = model.split("/")[0]
            username_path = models_dir / username_part
            sub_dirs = [d for d in username_path.iterdir() if d.is_dir()]
            if len(sub_dirs) == 1:
                model_path = username_path
    else:
        raise RemovalError(
            f"Model at {model_path} is not a valid .gguf file or safetensors model directory."
        )

    # Remove the specified file or directory.
    try:
        if model_path.is_file():
            logger.debug(f"Removing model file: {model}.")
            model_path.unlink()
        elif model_path.is_dir():
            logger.debug(f"Removing model directory: {model}.")
            shutil.rmtree(model_path)
        logger.debug(f"Model {model} has been removed.")

        # Remove cache if exists
        oci_dir = Path(DEFAULTS.OCI_DIR) / cache_model_path
        if oci_dir.exists():
            shutil.rmtree(oci_dir)
            logger.debug(f"Cache for model '{model}' has been removed.")
    except OSError as e:
        raise RemovalError(f"Error while trying to remove {model}: {e}") from e
