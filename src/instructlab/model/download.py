# SPDX-License-Identifier: Apache-2.0

# Standard
import abc
import json
import os
import re
import subprocess

# Third Party
from huggingface_hub import hf_hub_download, list_repo_files
from huggingface_hub import logging as hf_logging
from huggingface_hub import snapshot_download
import click

# First Party
from instructlab import clickext
from instructlab.configuration import DEFAULTS
from instructlab.utils import is_huggingface_repo, is_oci_repo


class Downloader(abc.ABC):
    """Base class for a downloading backend"""

    def __init__(
        self,
        repository: str,
        release: str,
        download_dest: str,
    ) -> None:
        self.repository = repository
        self.release = release
        self.download_dest = download_dest

    @abc.abstractmethod
    def download(self) -> None:
        """Downloads model from specified repo/release and stores it into download_dest"""


class HFDownloader(Downloader):
    """Class to handle downloading safetensors and GGUF models from Huggingface"""

    def __init__(
        self,
        repository: str,
        release: str,
        download_dest: str,
        filename: str,
        hf_token: str,
        ctx,
    ) -> None:
        super().__init__(
            repository=repository, release=release, download_dest=download_dest
        )
        self.repository = repository
        self.release = release
        self.download_dest = download_dest
        self.filename = filename
        self.hf_token = hf_token
        self.ctx = ctx

    def download(self):
        """Download the model(s) to train"""
        click.echo(
            f"Downloading model from huggingface: {self.repository}@{self.release} to {self.download_dest}..."
        )

        if self.hf_token == "" and "instructlab" not in self.repository:
            raise ValueError(
                """HF_TOKEN var needs to be set in your environment to download HF Model.
                Alternatively, the token can be passed with --hf-token flag.
                The HF Token is used to authenticate your identity to the Hugging Face Hub."""
            )

        try:
            if self.ctx.obj is not None:
                hf_logging.set_verbosity(self.ctx.obj.config.general.log_level.upper())
            files = list_repo_files(repo_id=self.repository, token=self.hf_token)
            if any(".safetensors" in string for string in files):
                self.download_safetensors()
            else:
                self.download_gguf()

        except Exception as exc:
            click.secho(
                f"Downloading model failed with the following Hugging Face Hub error: {exc}",
                fg="red",
            )
            raise click.exceptions.Exit(1)

    def download_gguf(self) -> None:
        try:
            hf_hub_download(
                token=self.hf_token,
                repo_id=self.repository,
                revision=self.release,
                filename=self.filename,
                local_dir=self.download_dest,
            )

        except Exception as exc:
            click.secho(
                f"Downloading GGUF model failed with the following HuggingFace Hub error: {exc}",
                fg="red",
            )
            raise click.exceptions.Exit(1)

    def download_safetensors(self) -> None:
        try:
            if not os.path.exists(os.path.join(self.download_dest, self.repository)):
                os.makedirs(
                    name=os.path.join(self.download_dest, self.repository),
                    exist_ok=True,
                )
            snapshot_download(
                token=self.hf_token,
                repo_id=self.repository,
                revision=self.release,
                local_dir=os.path.join(self.download_dest, self.repository),
            )
        except Exception as exc:
            click.secho(
                f"Downloading safetensors model failed with the following HuggingFace Hub error: {exc}",
                fg="red",
            )
            raise click.exceptions.Exit(1)


class OCIDownloader(Downloader):
    """
    Class to handle downloading safetensors models from OCI Registries
    We are leveraging OCI v1.1 for this functionality
    """

    def __init__(self, repository: str, release: str, download_dest: str, ctx) -> None:
        super().__init__(
            repository=repository, release=release, download_dest=download_dest
        )
        self.repository = repository
        self.release = release
        self.download_dest = download_dest
        self.ctx = ctx

    def _build_oci_model_file_map(self, oci_model_path: str) -> dict:
        """
        Helper function to build a mapping between blob files and what they represent
        """
        index_hash = ""
        try:
            with open(f"{oci_model_path}/index.json", mode="r", encoding="UTF-8") as f:
                index_ref = json.load(f)
            match = re.search("sha256:(.*)", index_ref["manifests"][0]["digest"])

            if match:
                index_hash = match.group(1)
            else:
                click.echo(f"could not find hash for index file at: {oci_model_path}")
                raise click.exceptions.Exit(1)
        except FileNotFoundError as exc:
            raise ValueError(f"file not found: {oci_model_path}/index.json") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"could not read JSON file: {oci_model_path}/index.json"
            ) from exc
        except Exception as exc:
            raise ValueError("unexpected error occurred: {e}") from exc

        try:
            with open(
                f"{oci_model_path}/blobs/sha256/{index_hash}",
                mode="r",
                encoding="UTF-8",
            ) as f:
                index = json.load(f)
        except FileNotFoundError as exc:
            raise ValueError(f"file not found: {oci_model_path}/index.json") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"could not read JSON file: {oci_model_path}/index.json"
            ) from exc
        except Exception as exc:
            raise ValueError("unexpected error occurred: {e}") from exc

        title_ref = "org.opencontainers.image.title"
        oci_model_file_map = {}

        for layer in index["layers"]:
            match = re.search("sha256:(.*)", layer["digest"])

            if match:
                blob_name = match.group(1)
                oci_model_file_map[blob_name] = layer["annotations"][title_ref]

        return oci_model_file_map

    def download(self):
        click.echo(
            f"Downloading model from OCI registry: {self.repository}@{self.release} to {self.download_dest}..."
        )

        os.makedirs(self.download_dest, exist_ok=True)
        model_name = self.repository.split("/")[-1]
        oci_dir = f"{DEFAULTS.OCI_DIR}/{model_name}"
        os.makedirs(oci_dir, exist_ok=True)

        command = [
            "skopeo",
            "copy",
            f"{self.repository}:{self.release}",
            f"oci:{oci_dir}",
        ]
        if self.ctx.obj.config.general.log_level == "DEBUG":
            command.append("--debug")

        try:
            subprocess.run(command, check=True)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "skopeo not installed, but required to perform downloads from OCI registries. Exiting",
            ) from exc
        except Exception as e:
            raise ValueError(
                f"CalledProcessError: command exited with non-zero code: {e}"
            ) from e

        file_map = self._build_oci_model_file_map(oci_dir)

        for _, _, files in os.walk(f"{oci_dir}/blobs/sha256/"):
            for name in files:
                if name not in file_map:
                    continue
                dest = file_map[name]
                if not os.path.exists(os.path.join(self.download_dest, model_name)):
                    os.makedirs(
                        os.path.join(self.download_dest, model_name), exist_ok=True
                    )
                # unlink any existing version of the file
                if os.path.exists(os.path.join(self.download_dest, model_name, dest)):
                    os.unlink(os.path.join(self.download_dest, model_name, dest))

                # create hard link to files in cache, to avoid redownloading if model has been downloaded before
                os.link(
                    os.path.join(f"{oci_dir}/blobs/sha256/", name),
                    os.path.join(self.download_dest, model_name, dest),
                )


@click.command()
@click.option(
    "--repository",
    default=DEFAULTS.MERLINITE_GGUF_REPO,  # TODO: add to config.yaml
    show_default=True,
    help="HuggingFace or OCI repository of the model to download.",
)
@click.option(
    "--release",
    default="main",  # TODO: add to config.yaml
    show_default=True,
    help="The revision of the model to download - e.g. a branch, tag, or commit hash for Huggingface repositories and tag or commit has for OCI repositories.",
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
@clickext.display_params
def download(ctx, repository, release, filename, model_dir, hf_token):
    downloader = None

    if is_oci_repo(repository):
        downloader = OCIDownloader(
            repository=repository, release=release, download_dest=model_dir, ctx=ctx
        )
    elif is_huggingface_repo(repository):
        downloader = HFDownloader(
            repository=repository,
            release=release,
            download_dest=model_dir,
            filename=filename,
            hf_token=hf_token,
            ctx=ctx,
        )
    else:
        click.secho(
            f"repository {repository} matches neither Huggingface, nor OCI registry format. Please supply a valid repository",
            fg="red",
        )
        raise click.exceptions.Exit(1)

    try:
        downloader.download()
    except Exception as exc:
        click.secho(
            f"Downloading model failed with the following error: {exc}",
            fg="red",
        )
        raise click.exceptions.Exit(1)
