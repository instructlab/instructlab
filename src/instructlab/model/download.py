# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
from typing import List
import abc
import logging
import os
import re
import subprocess

# Third Party
from huggingface_hub import hf_hub_download, list_repo_files
from huggingface_hub import logging as hf_logging
from huggingface_hub import snapshot_download

# First Party
from instructlab.configuration import DEFAULTS
from instructlab.defaults import DEFAULT_INDENT
from instructlab.utils import (
    check_skopeo_version,
    is_huggingface_repo,
    is_oci_repo,
    list_models,
    load_json,
    print_table,
)

logger = logging.getLogger(__name__)


class ModelDownloader(abc.ABC):
    """Base class for a downloading backend"""

    def __init__(
        self,
        log_level,
        repository: str,
        release: str,
        download_dest: Path,
    ) -> None:
        self.log_level = log_level
        self.repository = repository
        self.release = release
        self.download_dest = download_dest

    @abc.abstractmethod
    def download(self) -> None:
        """Downloads model from specified repo/release and stores it into download_dest"""


class HFDownloader(ModelDownloader):
    """Class to handle downloading safetensors and GGUF models from Hugging Face"""

    def __init__(
        self,
        repository: str,
        release: str,
        download_dest: Path,
        filename: str,
        hf_token: str,
        log_level: str,
    ) -> None:
        super().__init__(
            log_level=log_level,
            repository=repository,
            release=release,
            download_dest=download_dest,
        )
        self.filename = filename
        self.hf_token = hf_token or None

    def download(self):
        """
        Download specified model from Hugging Face
        """
        logger.info(
            f"Downloading model from Hugging Face:\n{DEFAULT_INDENT}Model: {self.repository}@{self.release}\n{DEFAULT_INDENT}Destination: {self.download_dest}"
        )

        if self.hf_token is None and "instructlab" not in self.repository:
            raise ValueError(
                """HF_TOKEN var needs to be set in your environment to download HF Model.
                Alternatively, the token can be passed with --hf-token flag.
                The HF Token is used to authenticate your identity to the Hugging Face Hub."""
            )

        try:
            if self.log_level is not None:
                hf_logging.set_verbosity(self.log_level)
            files = list_repo_files(repo_id=self.repository, token=self.hf_token)
            if any(re.search(r"\.(safetensors|bin)$", fname) for fname in files):
                self.download_entire_hf_repo()
            else:
                self.download_gguf()

        except Exception as exc:
            raise RuntimeError(
                f"\nDownloading model failed with the following Hugging Face Hub error:\n{DEFAULT_INDENT}{exc}"
            ) from exc

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
            raise RuntimeError(
                f"\nDownloading GGUF model failed with the following Hugging Face Hub error:\n{DEFAULT_INDENT}{exc}"
            ) from exc

    def download_entire_hf_repo(self) -> None:
        try:
            local_dir = os.path.join(self.download_dest, self.repository)
            os.makedirs(name=local_dir, exist_ok=True)

            snapshot_download(
                token=self.hf_token,
                repo_id=self.repository,
                revision=self.release,
                local_dir=local_dir,
            )
        except Exception as exc:
            raise RuntimeError(
                f"\nDownloading safetensors model failed with the following Hugging Face Hub error:\n{DEFAULT_INDENT}{exc}"
            ) from exc


class OCIDownloader(ModelDownloader):
    """
    Class to handle downloading safetensors models from OCI Registries
    We are leveraging OCI v1.1 for this functionality
    """

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
                raise KeyError(
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
            raise RuntimeError(
                f"\nFailed to build OCI model file mapping from:\n{DEFAULT_INDENT}{blob_dir}/{index_hash}"
            ) from exc

        return oci_model_file_map

    def download(self):
        logger.info(
            f"Downloading model from OCI registry:\n{DEFAULT_INDENT}Model: {self.repository}@{self.release}\n{DEFAULT_INDENT}Destination: {self.download_dest}"
        )

        # raise an exception if user specified tag/SHA embedded in repository instead of specifying --release
        match = re.search(r"^(?:[^:]*:){2}(.*)$", self.repository)
        if match:
            raise ValueError(
                f"\nInvalid repository supplied:\n{DEFAULT_INDENT}Please specify tag/version '{match.group(1)}' via --release"
            )

        model_name = self.repository.split("/")[-1]
        os.makedirs(os.path.join(self.download_dest, model_name), exist_ok=True)
        oci_dir = f"{DEFAULTS.OCI_DIR}/{model_name}"
        os.makedirs(oci_dir, exist_ok=True)

        # Check if skopeo is installed and the version is at least 1.9
        check_skopeo_version()

        command = [
            "skopeo",
            "copy",
            f"{self.repository}:{self.release}",
            f"oci:{oci_dir}",
            "--remove-signatures",
        ]
        if self.log_level == "DEBUG" and logger.isEnabledFor(logging.DEBUG):
            command.append("--debug")
        logger.debug(f"Running skopeo command: {command}")

        try:
            subprocess.run(command, check=True, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"\nFailed to run skopeo command:\n{DEFAULT_INDENT}{e}.\n{DEFAULT_INDENT}stderr: {e.stderr}"
            ) from e

        file_map = self._build_oci_model_file_map(oci_dir)
        if not file_map:
            raise LookupError("\nFailed to find OCI image blob hashes.")

        blob_dir = f"{oci_dir}/blobs/sha256/"
        for name, dest in file_map.items():
            dest_model_path = Path(self.download_dest) / model_name / str(dest)
            # unlink any existing version of the file
            if dest_model_path.exists():
                dest_model_path.unlink()

            if not dest_model_path.parent.exists():
                dest_model_path.parent.mkdir(parents=True)

            # create symlink to files in cache to avoid redownloading if model has been downloaded before
            os.symlink(
                os.path.join(blob_dir, name),
                dest_model_path,
            )


def download_models(
    log_level: str,
    repositories: List[str],
    releases: List[str],
    filenames: List[str],
    model_dir: Path,
    hf_token: str,
):
    """Downloads model from a specified repository"""
    downloader: ModelDownloader

    # strict = false ensures that if you just give --repository <some_safetensor> we won't error because len(filenames) is greater due to the defaults
    for repository, filename, release in zip(
        repositories, filenames, releases, strict=False
    ):
        if is_oci_repo(repository):
            downloader = OCIDownloader(
                log_level=log_level,
                repository=repository,
                release=release,
                download_dest=model_dir,
            )
        elif is_huggingface_repo(repository):
            downloader = HFDownloader(
                log_level=log_level,
                repository=repository,
                release=release,
                download_dest=model_dir,
                filename=filename,
                hf_token=hf_token,
            )
        else:
            raise ValueError(
                f"repository {repository} matches neither Hugging Face nor OCI registry format.\nPlease supply a valid repository"
            )

        try:
            downloader.download()
            logger.info(
                f"\nᕦ(òᴗóˇ)ᕤ {downloader.repository} model download completed successfully! ᕦ(òᴗóˇ)ᕤ\n"
            )
        # pylint: disable=broad-exception-caught
        except (ValueError, Exception) as exc:
            if isinstance(exc, ValueError) and "HF_TOKEN" in str(exc):
                logger.warning(
                    f"\n{downloader.repository} requires a HF Token to be set.\nPlease use '--hf-token' or 'export HF_TOKEN' to download all necessary models."
                )
            else:
                raise ValueError(
                    f"\nDownloading model failed with the following error: {exc}"
                ) from exc

    logger.info("Available models (`ilab model list`):")
    data = list_models([Path(model_dir)], False)
    data_as_lists = [list(item) for item in data]
    print_table(["Model Name", "Last Modified", "Size"], data_as_lists)
