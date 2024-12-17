# SPDX-License-Identifier: Apache-2.0

# Standard
import logging

# Third Party
import click

# First Party
from instructlab import clickext
from instructlab.defaults import UPLOAD_DESTINATIONS
from instructlab.model.upload import HFModelUploader

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--model",
    "-m",
    required=True,
    help="Path to the model to upload or name of an existing checkpoint.",
)
@click.option(
    "--dest-type",
    type=click.Choice(tuple(UPLOAD_DESTINATIONS)),
    default="hf",
    help="The type of destination to upload to - can be 'hf', 'oci', or 's3'",
)
@click.option(
    "--destination",
    "-d",
    type=str,
    required=True,
    help="Destination for the model to be uploaded to. Ex: for a Hugging Face repo, should be '<username>/<repo-name>'",
)
@click.option(
    "--release",
    "-rl",
    default="main",
    show_default=True,
    help="The revision to upload the model to - e.g. a branch for Hugging Face repositories.",
)
@click.option(
    "--hf-token",
    default="",
    envvar="HF_TOKEN",
    help="User access token for connecting to the Hugging Face Hub.",
)
@clickext.display_params
def upload(model, dest_type, destination, release, hf_token):
    """Uploads model to a specified location"""
    uploader = None

    if dest_type == "hf":
        uploader = HFModelUploader(
            model=model,
            destination=destination,
            release=release,
            hf_token=hf_token,
        )
    elif dest_type == "oci":
        # TODO: update this when OCI uploading is supported
        click.secho(
            f"Uploading of type {dest_type} is not yet supported",
            fg="yellow",
        )
        raise click.exceptions.Exit(1)
    elif dest_type == "s3":
        # TODO: update this when S3 uploading is supported
        click.secho(
            f"Uploading of type {dest_type} is not yet supported",
            fg="yellow",
        )
        raise click.exceptions.Exit(1)
    else:
        click.secho(
            f"{dest_type} matches neither Hugging Face, OCI registry, or an S3-compatable format.\nPlease supply a supported dest_type",
            fg="red",
        )
        raise click.exceptions.Exit(1)

    try:
        uploader.upload()
    # pylint: disable=broad-exception-caught
    except (ValueError, Exception) as exc:
        if isinstance(exc, ValueError) and "HF_TOKEN" in str(exc):
            click.secho(
                "Uploading to Hugging Face requires a HF Token to be set.\nPlease use '--hf-token' or 'export HF_TOKEN' to upload all necessary models.",
                fg="yellow",
            )
            raise click.exceptions.Exit(1)
        else:
            click.secho(
                f"\nUploading model failed with the following error: {exc}",
                fg="red",
            )
            raise click.exceptions.Exit(1)
