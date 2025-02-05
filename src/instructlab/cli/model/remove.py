# SPDX-License-Identifier: Apache-2.0

# Standard

# Third Party
import click

# First Party
from instructlab import clickext
from instructlab.configuration import DEFAULTS
from instructlab.model.remove import RemovalError, remove_model


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
def remove(model: str, force: bool, model_dir: str):
    """Remove model"""
    try:
        if force or click.confirm(
            f"Are you sure you want to remove the model '{model}'?", default=False
        ):
            click.echo(f"Removing model: {model}.")
            remove_model(model, model_dir)
            click.echo(f"Model {model} has been removed.")
        else:
            click.secho("Aborted deletion.", fg="yellow")
    except RemovalError as e:
        click.secho(str(e), fg="red")
        raise click.exceptions.Exit(1)
