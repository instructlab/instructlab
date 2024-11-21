# SPDX-License-Identifier: Apache-2.0

# Standard
from abc import ABC, abstractmethod
from pathlib import Path
import hashlib
import json
import logging
import os
import re
import subprocess

# Third Party
from huggingface_hub import hf_hub_download
from huggingface_hub import logging as hf_logging
from huggingface_hub import model_info, snapshot_download
from packaging import version
import click

# First Party
from instructlab import clickext
from instructlab.configuration import DEFAULTS
from instructlab.defaults import DEFAULT_INDENT
from instructlab.utils import load_json

logger = logging.getLogger(__name__)


class ModelMetadata:
    """Definition for metadata that is stored for downloaded models"""

    def __init__(self, v: str = "", SHA: str = "", size: int = 0) -> None:
        self.version = v
        self.SHA = SHA
        self.size = size


class DownloadTarget:
    """Parameters for model being targeted for download"""

    def __init__(
        self, repository: str = "", filename: str | None = None, release: str = "main"
    ) -> None:
        self.repository = repository
        self.filename = filename
        self.release = release


DEFAULT_TARGETS = [
    DownloadTarget(
        DEFAULTS.GRANITE_GGUF_REPO, DEFAULTS.GRANITE_GGUF_MODEL_NAME, "main"
    ),
    DownloadTarget(
        DEFAULTS.MERLINITE_GGUF_REPO, DEFAULTS.MERLINITE_GGUF_MODEL_NAME, "main"
    ),
    DownloadTarget(
        DEFAULTS.MISTRAL_GGUF_REPO, DEFAULTS.MISTRAL_GGUF_MODEL_NAME, "main"
    ),
]


class Downloader(ABC):
    """Base class for a downloading backend"""

    def __init__(
        self,
        ctx,
        repository: str,
        release: str,
        download_dest: str,
    ) -> None:
        self.ctx = ctx
        self.repository = repository
        self.release = release
        self.download_dest = download_dest
        self.metadata = ModelMetadata()
        self.metadata_filename = "metadata.json"

    @abstractmethod
    def download(self) -> None:
        """Downloads model from specified repo/release and stores it into download_dest"""

    @abstractmethod
    def get_metadata(self) -> None:
        """Retrieves metadata for model to be downloaded"""

    def dump_metadata(self, dest: Path) -> None:
        """Dumps metadata for downloaded model onto disk"""
        try:
            with open(dest / self.metadata_filename, "w", encoding="utf-8") as f:
                json.dump(self.metadata.__dict__, f)
        except OSError as e:
            # not blocking on failure to dump model metadata to disk
            logger.error(f"Failed to dump model metadata for {dest}: {e}")


class HFDownloader(Downloader):
    """Class to handle downloading safetensors and GGUF models from Hugging Face"""

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
            ctx=ctx, repository=repository, release=release, download_dest=download_dest
        )
        self.filename = filename
        self.hf_token = hf_token
        self.hf_prefix = "huggingface.co"

    @staticmethod
    def _is_huggingface_repo(repo_name: str) -> tuple[bool, str]:
        """
        Checks if a provided repository follows the huggingface URL syntax, and extracts
        the repository path if it does
        """
        # allow alphanumerics, underscores, hyphens and periods in huggingface repo names
        # repo name should be of the format <owner>/<model> or huggingface.co/<owner>/<model>
        pattern = r"^(huggingface\.co\/)?([\w.-]+\/[\w.-]+)$"
        match = re.search(pattern, repo_name)
        if match:
            return True, match.group(2)
        return False, ""

    def get_metadata(self) -> None:
        self.metadata.version = self.release

        modelInfo = model_info(
            repo_id=self.repository,
            revision=self.release,
            files_metadata=True,
        )

        # store commit SHA
        self.metadata.SHA = modelInfo.sha

        # calculate and store size in bytes so it can be used for system operations easily
        model_size = 0
        for sibling in modelInfo.siblings:
            # if a filename is specified take that file's size (gguf model)
            # else add up sizes of all files in the repo
            if self.filename is not None and self.filename == sibling.rfilename:
                model_size = sibling.size
                break
            model_size += sibling.size

        self.metadata.size = model_size

    def download(self):
        """
        Download specified model from Hugging Face
        """
        # allow/encourage users to prefix HF model URLs with "huggingface.co" for consistency with other
        # model source URLs and to align with the storage directory structure, but strip it off before making HF API calls
        _, self.repository = HFDownloader._is_huggingface_repo(self.repository)
        self.release = self.release if self.release != "" else "main"
        self.get_metadata()

        destination = os.path.join(
            self.download_dest, self.hf_prefix, self.repository, self.release
        )

        click.echo(
            f"Downloading model from Hugging Face:\n{DEFAULT_INDENT}Model: {self.repository}@{self.release}\n{DEFAULT_INDENT}Destination: {destination}"
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
            if self.filename != "":
                self.download_gguf(destination)
            else:
                self.download_entire_hf_repo(destination)

            super().dump_metadata(Path(destination))

        except Exception as exc:
            click.secho(
                f"\nDownloading model failed with the following Hugging Face Hub error:\n{DEFAULT_INDENT}{exc}",
                fg="red",
            )
            raise click.exceptions.Exit(1)

    def download_gguf(self, destination: str) -> None:
        try:
            hf_hub_download(
                token=self.hf_token,
                repo_id=self.repository,
                revision=self.release,
                filename=self.filename,
                local_dir=destination,
            )

        except Exception as exc:
            click.secho(
                f"\nDownloading GGUF model failed with the following Hugging Face Hub error:\n{DEFAULT_INDENT}{exc}",
                fg="red",
            )
            raise click.exceptions.Exit(1)

    def download_entire_hf_repo(self, destination: str) -> None:
        try:
            os.makedirs(name=destination, exist_ok=True)

            snapshot_download(
                token=self.hf_token,
                repo_id=self.repository,
                revision=self.release,
                local_dir=destination,
            )
        except Exception as exc:
            click.secho(
                f"\nDownloading safetensors model failed with the following Hugging Face Hub error:\n{DEFAULT_INDENT}{exc}",
                fg="red",
            )
            raise click.exceptions.Exit(1)


class OCIDownloader(Downloader):
    """
    Class to handle downloading safetensors models from OCI Registries
    We are leveraging OCI v1.1 for this functionality
    """

    def get_metadata(self) -> None:
        self.metadata.version = self.release

        # Check if skopeo is installed and the version is at least 1.9
        check_skopeo_version()

        # Run skopeo inspect to retrieve raw OCI manifest
        command = [
            "skopeo",
            "inspect",
            "--raw",
            f"{self.repository}:{self.release}",
        ]
        logger.debug(f"Running skopeo command: {command}")

        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError as e:
            click.secho(
                f"Failed to run skopeo command: {e}.\nstdout: {e.stdout}.\nstderr: {e.stderr}",
                fg="red",
            )
        manifest = result.stdout.decode("utf-8")

        # calculate and store the manifest digest
        self.metadata.SHA = hashlib.sha256(manifest.encode()).hexdigest()

        # calculate model size from layer information in manifest
        index = json.loads(manifest)
        model_size = 0
        for layer in index["layers"]:
            model_size += layer["size"]

        # storing size in bytes so it can be used for system operations easily
        self.metadata.size = model_size

    @staticmethod
    def _is_oci_repo(repo_url: str) -> tuple[bool, str]:
        """
        Checks if a provided repository follows the OCI registry URL syntax, and extracts
        the repository path if it does
        """

        oci_url_regex = r"^docker://([a-zA-Z0-9\-_.]+(:[0-9]+)?)(/[a-zA-Z0-9\-_.]+)+(@[a-zA-Z0-9]+|:[a-zA-Z0-9\-_.]+)?$"
        match = re.search(oci_url_regex, repo_url)
        if match:
            # group(0) is the full match, stripping the 'docker://' prefix
            full_repo_path = match.group(0)[9:]
            # Strip tag or digest if it exists (after ':' or '@')
            repo_without_tag = re.sub(
                r"(:[a-zA-Z0-9\-_.]+|@[a-zA-Z0-9]+)$", "", full_repo_path
            )
            return True, repo_without_tag
        return False, ""

    @staticmethod
    def _extract_sha(sha: str):
        return re.search("sha256:(.*)", sha)

    def _build_oci_model_file_map(self, oci_model_path: str) -> dict:
        """
        Helper function to build a mapping between blob files and what they represent
        Format for the index.json file can be found here: https://github.com/opencontainers/image-spec/blob/main/image-layout.md#indexjson-file
        """
        index_hash = ""
        index_ref_path = f"{oci_model_path}/index.json"
        try:
            index_ref = load_json(Path(index_ref_path))
            match = None
            for manifest in index_ref["manifests"]:
                if (
                    manifest["mediaType"]
                    == "application/vnd.oci.image.manifest.v1+json"
                ):
                    match = self._extract_sha(manifest["digest"])
                    break

            if match:
                index_hash = match.group(1)
            else:
                raise ValueError(
                    f"\nFailed to find hash in the index file:\n{DEFAULT_INDENT}{oci_model_path}"
                )
        except Exception as exc:
            raise ValueError(
                f"\nFailed to extract image hash from index file:\n{DEFAULT_INDENT}{oci_model_path}"
            ) from exc

        blob_dir = f"{oci_model_path}/blobs/sha256"
        index = load_json(Path(f"{blob_dir}/{index_hash}"))

        title_ref = "org.opencontainers.image.title"
        oci_model_file_map = {}
        try:
            for layer in index["layers"]:
                match = self._extract_sha(layer["digest"])

                if match:
                    blob_name = match.group(1)
                    oci_model_file_map[blob_name] = layer["annotations"][title_ref]
        except Exception as exc:
            raise ValueError(
                f"\nFailed to build OCI model file mapping from:\n{DEFAULT_INDENT}{blob_dir}/{index_hash}"
            ) from exc

        return oci_model_file_map

    def download(self):
        self.get_metadata()

        # Try to isolate the path to where the model is stored via the URL and replicate that structure locally
        # If unable to do that for whatever reason, just use the model name and create a folder out of that
        model_sub_dirs = []
        _, repo = OCIDownloader._is_oci_repo(self.repository)
        if repo != "":
            model_sub_dirs = repo.split("/")
            model_dest_dir = os.path.join(
                self.download_dest, *model_sub_dirs, self.release
            )
            oci_dir = os.path.join(DEFAULTS.OCI_DIR, *model_sub_dirs, self.release)
        else:
            model_name = self.repository.split("/")[-1]
            model_dest_dir = os.path.join(self.download_dest, model_name, self.release)
            oci_dir = f"{DEFAULTS.OCI_DIR}/{model_name}/{self.release}"
        os.makedirs(model_dest_dir, exist_ok=True)
        os.makedirs(oci_dir, exist_ok=True)

        dest_model_path = os.path.join(
            self.download_dest, *model_sub_dirs, self.release
        )

        click.echo(
            f"Downloading model from OCI registry:\n{DEFAULT_INDENT}Model: {self.repository}@{self.release}\n{DEFAULT_INDENT}Destination: {dest_model_path}"
        )

        command = [
            "skopeo",
            "copy",
            f"{self.repository}:{self.release}",
            f"oci:{oci_dir}",
            "--remove-signatures",
        ]
        if self.ctx.obj is not None and logger.isEnabledFor(logging.DEBUG):
            command.append("--debug")
        logger.debug(f"Running skopeo command: {command}")

        try:
            subprocess.run(command, check=True, text=True)
        except subprocess.CalledProcessError as e:
            click.secho(
                f"\nFailed to run skopeo command:\n{DEFAULT_INDENT}{e}.\n{DEFAULT_INDENT}stderr: {e.stderr}",
                fg="red",
            )
            raise click.exceptions.Exit(1)

        file_map = self._build_oci_model_file_map(oci_dir)
        if not file_map:
            click.secho(
                "\nFailed to find OCI image blob hashes.",
                fg="red",
            )
            raise click.exceptions.Exit(1)

        blob_dir = f"{oci_dir}/blobs/sha256/"

        for name, dest in file_map.items():
            model_file = Path(dest_model_path) / str(dest)
            # unlink any existing version of the file
            if model_file.exists():
                model_file.unlink()

            if not model_file.parent.exists():
                model_file.parent.mkdir(parents=True)

            # create symlink to files in cache to avoid redownloading if model has been downloaded before
            os.symlink(
                os.path.join(blob_dir, name),
                model_file,
            )

        super().dump_metadata(Path(dest_model_path))


@click.command()
@click.option(
    "--repository",
    "-rp",
    "repositories",
    multiple=True,
    show_default=True,
    help="Hugging Face or OCI repository of the model to download.",
)
@click.option(
    "--release",
    "-rl",
    "releases",
    multiple=True,
    show_default=True,
    help="The revision of the model to download - e.g. a branch, tag, or commit hash for Hugging Face repositories and tag or commit hash for OCI repositories.",
)
@click.option(
    "--filename",
    "filenames",
    multiple=True,
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
def download(ctx, repositories, releases, filenames, model_dir, hf_token):
    """Downloads model from a specified repository"""
    downloader = None
    targets = []

    # fill in defaults only when user passes no arguments to `ilab model download`
    if not repositories and not filenames and not releases:
        targets = DEFAULT_TARGETS
    else:
        for idx, repo in enumerate(repositories):
            targets.append(
                DownloadTarget(
                    repo,
                    filenames[idx] if idx < len(filenames) else "",
                    releases[idx] if idx < len(releases) else "",
                )
            )

    for tgt in targets:
        # raise an exception if user specified tag/SHA embedded in repository that conflicts with --release
        match = re.search(r"(.*):(\w+)$", tgt.repository)
        if match:
            if tgt.release != "" and tgt.release != match.group(2):
                click.secho(
                    f"Conflicting versions supplied: '{match.group(2)}' and {tgt.release}. Pkease specify one or the other.",
                    fg="red",
                )
                raise click.exceptions.Exit(1)
            tgt.release = match.group(2)
            tgt.repository = match.group(1)

        if OCIDownloader._is_oci_repo(tgt.repository)[0]:
            downloader = OCIDownloader(
                ctx=ctx,
                repository=tgt.repository,
                release=tgt.release,
                download_dest=model_dir,
            )
        elif HFDownloader._is_huggingface_repo(tgt.repository)[0]:
            downloader = HFDownloader(
                ctx=ctx,
                repository=tgt.repository,
                release=tgt.release,
                download_dest=model_dir,
                filename=tgt.filename,
                hf_token=hf_token,
            )
        else:
            click.secho(
                f"repository {tgt.repository} matches neither Hugging Face nor OCI registry format.\nPlease supply a valid repository",
                fg="red",
            )
            raise click.exceptions.Exit(1)

        try:
            downloader.download()
        # pylint: disable=broad-exception-caught
        except (ValueError, Exception) as exc:
            if isinstance(exc, ValueError) and "HF_TOKEN" in str(exc):
                click.secho(
                    f"\n{downloader.repository} requires a HF Token to be set.\nPlease use '--hf-token' or 'export HF_TOKEN' to download all necessary models.",
                    fg="yellow",
                )
            else:
                click.secho(
                    f"\nDownloading model failed with the following error: {exc}",
                    fg="red",
                )
                raise click.exceptions.Exit(1)


_RECOMMENDED_SCOPEO_VERSION = "1.9.0"


def check_skopeo_version():
    """
    Check if skopeo is installed and the version is at least 1.9.0
    This is required for downloading models from OCI registries.
    """
    # Run the 'skopeo --version' command and capture the output
    try:
        result = subprocess.run(
            ["skopeo", "--version"], capture_output=True, text=True, check=True
        )
    except FileNotFoundError as exc:
        click.secho(
            f"\nskopeo is not installed.\nPlease install recommended version {_RECOMMENDED_SCOPEO_VERSION}",
            fg="red",
        )
        raise click.exceptions.Exit(1) from exc

    logger.debug(f"'skopeo --version' output: {result.stdout}")

    # Extract the version number using a regular expression
    match = re.search(r"skopeo version (\d+\.\d+\.\d+)", result.stdout)
    if match:
        installed_version = match.group(1)
        logger.debug(f"detected skopeo version: {installed_version}")

        # Compare the extracted version with the required version
        if version.parse(installed_version) < version.parse(
            _RECOMMENDED_SCOPEO_VERSION
        ):
            raise ValueError(
                f"\n{DEFAULT_INDENT}skopeo version {installed_version} is lower than {_RECOMMENDED_SCOPEO_VERSION}.\n{DEFAULT_INDENT}Please upgrade it."
            )
    else:
        logger.warning(
            f"Failed to determine skopeo version. Recommended version is {_RECOMMENDED_SCOPEO_VERSION}. Downloading the model might fail."
        )
