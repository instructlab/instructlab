# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
import shutil

# Third Party
import click

# First Party
from instructlab import clickext
from instructlab.configuration import DEFAULTS
from instructlab.model.backends.backends import is_model_gguf, is_model_safetensors


class RemovalError(Exception):
    """Custom exception for removal errors."""


def remove_item(model_path: Path, model_name: str):
    """Remove the specified file or directory."""
    try:
        if model_path.is_file():
            print(f"Removing model file: {model_name}.")
            model_path.unlink()
        elif model_path.is_dir():
            print(f"Removing model directory: {model_name}.")
            shutil.rmtree(model_path)
        print(f"Model {model_name} has been removed.")
    except OSError as e:
        raise RemovalError(f"Error while trying to remove {model_name}: {e}") from e


@click.command(name="remove")
@clickext.display_params
@click.option(
    "-m",
    "--model",
    required=True,
    help="Specify the model name to remove (check with 'ilab model list' first), e.g., merlinite-7b-lab-Q4_K_M.gguf or instructlab/merlinite-7b-pt.",
)
@click.option(
    "-f",
    "--force",
    is_flag=True,
    help="Used to forcefully remove a model. Use with caution as it permanently deletes the model from the storage directory without further confirmation.",
)
@click.option(
    "-md",
    "--model-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=DEFAULTS.MODELS_DIR,
    help=f"Specify the base directory for models. [default: {DEFAULTS.MODELS_DIR}]",
)
def remove_model(model: str, force: bool, model_dir: str):
    """Remove model"""
    models_dir = Path(model_dir)
    model_path = models_dir / model
    base_dir_str = Path(model_dir).name + "/"

    # Without <username> model dir and start with models/
    if model.startswith(base_dir_str):
        remove_models_path = model[len(base_dir_str) :]
        model_path = models_dir / remove_models_path

    if not model_path.exists():
        click.secho(
            f"Error: Model {model} does not exist in {models_dir}.",
            fg="red",
        )
        raise click.exceptions.Exit(1)

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
        click.secho(
            f"Error: Model found at {model_path} is not a valid .gguf file or safetensors model directory.",
            fg="red",
        )
        raise click.exceptions.Exit(1)

    try:
        if force or click.confirm(
            f"Are you sure you want to remove the model '{model}'?", default=False
        ):
            remove_item(model_path, model)
        else:
            click.secho("Aborted deletion.", fg="yellow")
    except RemovalError as e:
        click.secho(str(e), fg="red")
        raise click.exceptions.Exit(1)
