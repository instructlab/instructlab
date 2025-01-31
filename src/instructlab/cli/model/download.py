# Standard
from pathlib import Path

# Third Party
import click

# First Party
from instructlab import clickext
from instructlab.defaults import DEFAULTS
from instructlab.model.download import download_models


@click.command()
@click.option(
    "--repository",
    "-rp",
    "repositories",
    multiple=True,
    default=[
        DEFAULTS.GRANITE_GGUF_REPO,
        DEFAULTS.MERLINITE_GGUF_REPO,
        DEFAULTS.MISTRAL_GGUF_REPO,
    ],  # TODO: add to config.yaml
    show_default=True,
    help="Hugging Face, OCI, or S3 repository of the model to download.",
)
@click.option(
    "--release",
    "-rl",
    "releases",
    multiple=True,
    default=[
        "main",
        "main",
        "main",
    ],  # TODO: add to config.yaml
    show_default=True,
    help="The revision of the model to download - e.g. a branch, tag, or commit hash for Hugging Face repositories and tag or commit hash for OCI repositories.",
)
@click.option(
    "--filename",
    "filenames",
    multiple=True,
    default=[
        DEFAULTS.GRANITE_GGUF_MODEL_NAME,
        DEFAULTS.MERLINITE_GGUF_MODEL_NAME,
        DEFAULTS.MISTRAL_GGUF_MODEL_NAME,
    ],
    show_default="The default model location in the instructlab data directory.",
    help="Name of the model file to download from the Hugging Face or S3 repository.",
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
@clickext.display_params
def download(ctx, repositories, releases, filenames, model_dir, hf_token):
    """Downloads models to a specified location"""

    try:
        model = Path(model_dir)
        download_models(
            log_level=ctx.obj.config.general.log_level.upper(),
            repositories=repositories,
            releases=releases,
            filenames=filenames,
            model_dir=model,
            hf_token=hf_token,
        )
    except Exception as e:
        click.secho(f"Downloading failed with the following exception: {e}", fg="red")
        raise click.exceptions.Exit(1)
