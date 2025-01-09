# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
import abc
import logging
import os
import re
import subprocess

# Third Party
from huggingface_hub import HfApi

# First Party
from instructlab.configuration import DEFAULTS
from instructlab.defaults import DEFAULT_INDENT
from instructlab.utils import (
    check_skopeo_version,
    is_model_gguf,
    is_model_safetensors,
    is_oci_repo,
)

logger = logging.getLogger(__name__)


class InvalidModelForUpload(Exception):
    """
    Error raised when a model is not able to be uploaded

    Attributes
        path        filepath of model location
    """

    def __init__(self, path) -> None:
        self.path = path
        super().__init__(
            f"Local model path {path} is a valid path, but is not a compliant format - cannot upload"
        )


class InvalidDestinationForUpload(Exception):
    """
    Error raised when a destination is not able to be uploaded to

    Attributes
        destination        destination of upload
        release            release of upload

    """

    def __init__(self, destination, release) -> None:
        self.destination = destination
        self.release = release
        super().__init__(f"Cannot upload model to {destination}@{release}")


class HFUploadError(Exception):
    """
    Error raised when an InstructLab call to Hugging Face for model uploading fails

    Attributes
        path        filepath of model location
        model_type  Type of model being uploaded, i.e. safetensors or GGUF
        hf_exc      Hugging Face exception
    """

    def __init__(self, path, model_type, hf_exc) -> None:
        self.path = path
        self.model_type = model_type
        self.hf_exc = hf_exc
        super().__init__(
            f"Uploading {self.model_type} model at {self.path} failed with the following Hugging Face Hub error:\n{self.hf_exc}"
        )


class OCIUploadError(Exception):
    """
    Error raised when an InstructLab call to an OCI registry for model uploading fails

    Attributes
        path        filepath of model location
        oci_exc     OCI registry exception
    """

    def __init__(self, path, oci_exc) -> None:
        self.path = path
        self.oci_exc = oci_exc
        super().__init__(
            f"Uploading OCI model at {self.path} failed with the following error:\n{self.oci_exc}"
        )


class ModelUploader(abc.ABC):
    """Base class for a model uploading backend"""

    def __init__(
        self,
        model: str,
        destination: str,
        release: str,
    ) -> None:
        self.model = model
        self.destination = destination
        self.release = release

    @abc.abstractmethod
    def upload(self) -> None:
        """Uploads specified local model and stores it into destination@release"""


class HFModelUploader(ModelUploader):
    """Class to handle uploading safetensors/bin and GGUF models to Hugging Face"""

    def __init__(
        self,
        model: str,
        destination: str,
        release: str,
        hf_token: str,
    ) -> None:
        super().__init__(
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

        logger.info(
            f"Uploading model to Hugging Face:\n{DEFAULT_INDENT}Model: {self.model}\n{DEFAULT_INDENT}Destination: {self.destination}@{self.release}\n"
        )

        # determine if model is path or name - look in checkpoints dir if the latter
        if os.path.exists(self.model):
            self.local_model_path = Path(self.model)
            self.model = self.model.split("/")[-1]
        else:
            self.local_model_path = os.path.join(DEFAULTS.CHECKPOINTS_DIR, self.model)
            # throw an error if the built path is invalid
            if not os.path.exists(self.local_model_path):
                raise FileNotFoundError(
                    f"Couldn't find model at {self.local_model_path} - are you sure it exists?"
                )

        # upload either safetensors or gguf, if valid
        if is_model_safetensors(self.local_model_path):
            self.upload_safetensors()
        elif is_model_gguf(self.local_model_path):
            self.upload_gguf()
        else:
            raise InvalidModelForUpload(self.local_model_path)

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
            raise HFUploadError(self.local_model_path, "GGUF", exc) from exc
        logger.info(
            f"\nUploading GGUF model at {self.local_model_path} succeeded!",
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
            raise HFUploadError(self.local_model_path, "safetensors", exc) from exc
        logger.info(
            f"\nUploading safetensors model at {self.local_model_path} succeeded!"
        )


class OCIModelUploader(ModelUploader):
    """
    Class to handle uploading OCI-formatted models to an OCI registry endpoint
    We are leveraging OCI v1.1 for this functionality
    """

    def __init__(
        self,
        model: str,
        destination: str,
        release: str,
    ) -> None:
        super().__init__(
            model=model,
            destination=destination,
            release=release,
        )

    def upload(self):
        """
        Upload specified model to the specified OCI registry
        """

        logger.info(
            f"Uploading model to OCI registry:\n{DEFAULT_INDENT}Model: {self.model}\n{DEFAULT_INDENT}Destination: {self.destination}@{self.release}\n"
        )

        # determine if model is path or name - look in checkpoints dir if the latter
        if os.path.exists(self.model):
            self.local_model_path = Path(self.model)
            self.model = self.model.split("/")[-1]
        else:
            self.local_model_path = os.path.join(DEFAULTS.CHECKPOINTS_DIR, self.model)
            # throw an error if the built path is invalid
            if not os.path.exists(self.local_model_path):
                raise FileNotFoundError(
                    f"Couldn't find model at {self.local_model_path} - are you sure it exists?"
                )

        # raise an exception if user specified tag/SHA embedded in repository instead of specifying --release
        invalid_model = re.search(r"^(?:[^:]*:){2}(.*)$", str(self.local_model_path))
        if invalid_model:
            logger.error(
                f"\nInvalid model supplied:\n{DEFAULT_INDENT}Please specify tag/version '{invalid_model.group(1)}' via --release",
            )
            raise InvalidModelForUpload(self.local_model_path)
        if not is_oci_repo(self.destination):
            logger.error(
                f"\nInvalid destination supplied:\n{DEFAULT_INDENT}Please specify valid OCI repository URL syntax via --destination",
            )
            raise InvalidDestinationForUpload(self.destination, self.release)
        # upload OCI model, if valid
        self.upload_oci()

    def upload_oci(self) -> None:
        # Check if skopeo is installed and the version is at least 1.9
        check_skopeo_version()

        command = [
            "skopeo",
            "copy",
            f"oci:{self.local_model_path}",
            f"{self.destination}:{self.release}",
            "--remove-signatures",
        ]
        if logger.isEnabledFor(logging.DEBUG):
            command.append("--debug")
        logger.debug(f"Running skopeo command: {command}")

        try:
            subprocess.run(command, check=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise OCIUploadError(self.local_model_path, exc) from exc
        logger.info(
            f"Uploading OCI model at {self.local_model_path} succeeded!",
        )
