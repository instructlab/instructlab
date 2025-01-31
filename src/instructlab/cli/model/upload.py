# SPDX-License-Identifier: Apache-2.0

# Standard
import logging

# Third Party
import click

# First Party
from instructlab import clickext
from instructlab.defaults import UPLOAD_DESTINATIONS
from instructlab.model.upload import upload_model

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--model",
    "-m",
    type=str,
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
    help="The revision to upload the model to. Ex: a branch for Hugging Face repositories. Not needed for 's3' uploads.",
)
@click.option(
    "--hf-token",
    default="",
    envvar="HF_TOKEN",
    help="User access token for connecting to the Hugging Face Hub.",
)
@click.pass_context
@clickext.display_params
def upload(ctx, model, dest_type, destination, release, hf_token):
    """Uploads model to a specified location"""

    try:
        upload_model(
            log_level=ctx.obj.config.general.log_level.upper(),
            dest_type=dest_type,
            model=model,
            destination=destination,
            release=release,
            hf_token=hf_token,
        )
    except Exception as e:
        click.secho(f"Uploading failed with the following exception: {e}", fg="red")
        raise click.exceptions.Exit(1)
