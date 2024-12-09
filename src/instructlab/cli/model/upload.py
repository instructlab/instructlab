# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
import abc
import logging
import os

# Third Party
from huggingface_hub import HfApi
from huggingface_hub import logging as hf_logging
import click

# First Party
from instructlab import clickext
from instructlab.configuration import DEFAULTS
from instructlab.defaults import DEFAULT_INDENT
from instructlab.utils import is_model_gguf, is_model_safetensors

logger = logging.getLogger(__name__)


class ModelUploader(abc.ABC):
    """Base class for a model uploading backend"""

    def __init__(
        self,
        ctx,
        model: str,
        destination: str,
        release: str,
    ) -> None:
        self.ctx = ctx
        self.model = model
        self.destination = destination
        self.release = release

    @abc.abstractmethod
    def upload(self) -> None:
        """Uploads specified local model and stores it into destination@release"""


class HFModelUploader(ModelUploader):
    """Class to handle uploading safetensors and GGUF models to Hugging Face"""

    def __init__(
        self,
        ctx,
        model: str,
        destination: str,
        release: str,
        hf_token: str,
    ) -> None:
        super().__init__(
            ctx=ctx,
            model=model,
            destination=destination,
            release=release,
        )
        self.hf_token = (
            hf_token.strip()
        )  # Remove trailing whitespaces from the token if they exist
        self.local_model_path = ""

    def upload(self):
        """
        Upload specified model to Hugging Face
        """

        # HF token check
        if len(self.hf_token) == 0:
            raise ValueError(
                """HF_TOKEN var needs to be set in your environment to upload HF Model.
                Alternatively, the token can be passed with --hf-token flag.
                The HF Token is used to authenticate your identity to the Hugging Face Hub."""
            )

        click.echo(
            f"Uploading model to Hugging Face:\n{DEFAULT_INDENT}Model: {self.model}\n{DEFAULT_INDENT}Destination: {self.destination}@{self.release}\n"
        )

        # set logger
        if self.ctx.obj is not None:
            hf_logging.set_verbosity(self.ctx.obj.config.general.log_level.upper())

        # determine if model is path or name - look in checkpoints dir if the latter
        if os.path.exists(self.model):
            self.local_model_path = Path(self.model)
            self.model = self.model.split("/")[-1]
        else:
            self.local_model_path = os.path.join(DEFAULTS.CHECKPOINTS_DIR, self.model)
            # throw an error if the built path is invalid
            if not os.path.exists(self.local_model_path):
                click.secho(
                    f"Couldn't find model at {self.local_model_path} - are you sure it exists?",
                    fg="red",
                )
                raise click.exceptions.Exit(1)

        # upload either safetensors or gguf, if valid
        if is_model_safetensors(self.local_model_path):
            self.upload_safetensors()
        elif is_model_gguf(self.local_model_path):
            self.upload_gguf()
        else:
            click.secho(
                f"Local model path {self.local_model_path} is a valid path, but is neither safetensors nor a GGUF - cannot upload to Hugging Face",
                fg="red",
            )
            raise click.exceptions.Exit(1)

    def upload_gguf(self) -> None:
        try:
            hf_api = HfApi()
            resp = hf_api.upload_file(
                path_or_fileobj=self.local_model_path,
                path_in_repo=self.model,
                repo_id=self.destination,
                revision=self.release,
                token=self.hf_token,
            )
            logger.debug(f"Hugging Face response:\n{resp}")
        except Exception as exc:
            click.secho(
                f"\nUploading GGUF model at {self.local_model_path} failed with the following Hugging Face Hub error:\n{DEFAULT_INDENT}{exc}",
                fg="red",
            )
            raise click.exceptions.Exit(1)
        click.secho(
            f"\nUploading GGUF model at {self.local_model_path} succeeded!",
            fg="green",
        )

    def upload_safetensors(self) -> None:
        try:
            hf_api = HfApi()
            resp = hf_api.upload_folder(
                folder_path=self.local_model_path,
                repo_id=self.destination,
                revision=self.release,
                token=self.hf_token,
            )
            logger.debug(f"Hugging Face response:\n{resp}")
        except Exception as exc:
            click.secho(
                f"\nUploading safetensors model at {self.local_model_path} failed with the following Hugging Face Hub error:\n{DEFAULT_INDENT}{exc}",
                fg="red",
            )
            raise click.exceptions.Exit(1)
        click.secho(
            f"\nUploading safetensors model at {self.local_model_path} succeeded!",
            fg="green",
        )


@click.command()
@click.option(
    "--model",
    "-m",
    required=True,
    help="Path to the model to upload or name of an existing checkpoint.",
)
@click.option(
    "--dest-type",
    type=click.Choice(["hf", "oci", "s3"]),
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
@click.pass_context
@clickext.display_params
def upload(ctx, model, dest_type, destination, release, hf_token):
    """Uploads model to a specified location"""
    uploader = None

    if dest_type == "hf":
        uploader = HFModelUploader(
            ctx=ctx,
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
