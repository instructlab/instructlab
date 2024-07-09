# SPDX-License-Identifier: Apache-2.0

# Standard
import os

# Third Party
from huggingface_hub import hf_hub_download, list_repo_files
from huggingface_hub import logging as hf_logging
from huggingface_hub import snapshot_download
import click

# First Party
from instructlab import utils
from instructlab.configuration import DEFAULTS


@click.command()
@click.option(
    "--repository",
    default=DEFAULTS.MERLINITE_GGUF_REPO,  # TODO: add to config.yaml
    show_default=True,
    help="Hugging Face repository of the model to download.",
)
@click.option(
    "--release",
    default="main",  # TODO: add to config.yaml
    show_default=True,
    help="The git revision of the model to download - e.g. a branch, tag, or commit hash.",
)
@click.option(
    "--filename",
    default=DEFAULTS.GGUF_MODEL_NAME,
    show_default="The default model location in the instructlab data directory.",
    help="Name of the model file to download from the Hugging Face repository.",
)
@click.option(
    "--model-dir",
    default=lambda: DEFAULTS.MODELS_DIR,
    show_default="The default system model location store, located in the data directory.",
    help="The local directory to download the model files into.",
)
@click.option(
    "--hf-token",
    default="",
    envvar="HF_TOKEN",
    help="User access token for connecting to the Hugging Face Hub.",
)
@click.pass_context
@utils.display_params
def download(ctx, repository, release, filename, model_dir, hf_token):
    """Download the model(s) to train"""
    click.echo(f"Downloading model from {repository}@{release} to {model_dir}...")
    if hf_token == "" and "instructlab" not in repository:
        raise ValueError(
            """HF_TOKEN var needs to be set in your environment to download HF Model.
            Alternatively, the token can be passed with --hf-token flag.
            The HF Token is used to authenticate your identity to the Hugging Face Hub."""
        )
    try:
        if ctx.obj is not None:
            hf_logging.set_verbosity(ctx.obj.config.general.log_level.upper())
        files = list_repo_files(repo_id=repository, token=hf_token)
        if any(".safetensors" in string for string in files):
            if not os.path.exists(os.path.join(model_dir, repository)):
                os.makedirs(name=os.path.join(model_dir, repository), exist_ok=True)
            snapshot_download(
                token=hf_token,
                repo_id=repository,
                revision=release,
                local_dir=os.path.join(model_dir, repository),
            )
        else:
            hf_hub_download(
                token=hf_token,
                repo_id=repository,
                revision=release,
                filename=filename,
                local_dir=model_dir,
            )
    except Exception as exc:
        click.secho(
            f"Downloading model failed with the following Hugging Face Hub error: {exc}",
            fg="red",
        )
        raise click.exceptions.Exit(1)
