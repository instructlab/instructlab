# SPDX-License-Identifier: Apache-2.0

# Standard
from unittest.mock import Mock, patch
import pathlib
import struct

# Third Party
from click.testing import CliRunner
from gguf.constants import GGUF_MAGIC

# First Party
from instructlab import lab
from instructlab.configuration import DEFAULTS
from tests.test_backends import create_safetensors_or_bin_model_files


class TestLabUpload:
    # When using `from X import Y` you need to understand that Y becomes part
    # of your module, so you should use `my_module.Y`` to patch.
    # When using `import X`, you should use `X.Y` to patch.
    # https://docs.python.org/3/library/unittest.mock.html#where-to-patch?
    @patch("instructlab.model.upload.HfApi.upload_file")
    def test_upload_gguf_hf(
        self,
        mock_hf_hub_upload_file: Mock,  # pylint: disable=unused-argument
        tmp_path: pathlib.Path,
        cli_runner: CliRunner,
    ):
        tmp_gguf = tmp_path / "model.gguf"
        with open(tmp_gguf, "wb") as gguf_file:
            gguf_file.write(struct.pack("<I", GGUF_MAGIC))
        result = cli_runner.invoke(
            lab.ilab,
            [
                "--config=DEFAULT",
                "model",
                "upload",
                f"--model={tmp_gguf}",
                "--dest-type=hf",
                "--destination=testuser/testgguf",
                "--hf-token=foo",
            ],
        )
        assert (
            result.exit_code == 0
        ), f"command finished with an unexpected exit code. {result.stdout}"
        assert f"Uploading GGUF model at {tmp_gguf} succeeded!" in result.output

    @patch("instructlab.model.upload.HfApi.upload_folder")
    def test_upload_safetensors_hf(
        self,
        mock_hf_hub_upload_folder: Mock,  # pylint: disable=unused-argument
        tmp_path: pathlib.Path,
        cli_runner: CliRunner,
    ):
        tmp_safetensor_dir = tmp_path / "tmp_safetensor_model"
        create_safetensors_or_bin_model_files(tmp_safetensor_dir, "safetensors", True)
        result = cli_runner.invoke(
            lab.ilab,
            [
                "--config=DEFAULT",
                "model",
                "upload",
                f"--model={tmp_safetensor_dir}",
                "--dest-type=hf",
                "--destination=testuser/testgguf",
                "--hf-token=foo",
            ],
        )
        assert (
            result.exit_code == 0
        ), f"command finished with an unexpected exit code. {result.stdout}"
        assert (
            f"Uploading safetensors model at {tmp_safetensor_dir} succeeded!"
            in result.output
        )

    def test_upload_invalid_gguf_hf(
        self, tmp_path: pathlib.Path, cli_runner: CliRunner
    ):
        tmp_gguf = tmp_path / "model.gguf"
        with open(tmp_gguf, "w", encoding="utf-8") as gguf_file:
            gguf_file.write("")
        result = cli_runner.invoke(
            lab.ilab,
            [
                "--config=DEFAULT",
                "model",
                "upload",
                f"--model={tmp_gguf}",
                "--dest-type=hf",
                "--destination=testuser/testgguf",
                "--hf-token=foo",
            ],
        )
        assert (
            result.exit_code == 1
        ), f"command finished with an unexpected exit code. {result.stdout}"
        assert (
            f"Local model path {tmp_gguf} is a valid path, but is neither safetensors nor a GGUF - cannot upload"
        ) in result.output

    def test_upload_invalid_safetensors_hf(
        self, tmp_path: pathlib.Path, cli_runner: CliRunner
    ):
        tmp_safetensor_dir = tmp_path / "tmp_safetensor_model"
        create_safetensors_or_bin_model_files(tmp_safetensor_dir, "safetensors", False)
        result = cli_runner.invoke(
            lab.ilab,
            [
                "--config=DEFAULT",
                "model",
                "upload",
                f"--model={tmp_safetensor_dir}",
                "--dest-type=hf",
                "--destination=testuser/testgguf",
                "--hf-token=foo",
            ],
        )
        assert (
            result.exit_code == 1
        ), f"command finished with an unexpected exit code. {result.stdout}"
        assert (
            f"Local model path {tmp_safetensor_dir} is a valid path, but is neither safetensors nor a GGUF - cannot upload"
        ) in result.output

    def test_upload_bad_model_hf(self, cli_runner: CliRunner):
        tmp_gguf = "model.gguf"
        result = cli_runner.invoke(
            lab.ilab,
            [
                "--config=DEFAULT",
                "model",
                "upload",
                f"--model={tmp_gguf}",
                "--dest-type=hf",
                "--destination=testuser/testgguf",
                "--hf-token=foo",
            ],
        )
        assert (
            result.exit_code == 1
        ), f"command finished with an unexpected exit code. {result.stdout}"
        assert (
            f"Couldn't find model at {DEFAULTS.CHECKPOINTS_DIR}/{tmp_gguf} - are you sure it exists?"
            in result.output
        )

    def test_upload_no_token_hf(self, tmp_path: pathlib.Path, cli_runner: CliRunner):
        tmp_gguf = tmp_path / "model.gguf"
        result = cli_runner.invoke(
            lab.ilab,
            [
                "--config=DEFAULT",
                "model",
                "upload",
                f"--model={tmp_gguf}",
                "--dest-type=hf",
                "--destination=testuser/testgguf",
            ],
        )
        assert (
            result.exit_code == 1
        ), f"command finished with an unexpected exit code. {result.stdout}"
        assert (
            "Uploading to Hugging Face requires a HF Token to be set.\nPlease use '--hf-token' or 'export HF_TOKEN' to upload all necessary models."
            in result.output
        )

    def test_upload_bad_dest_hf(self, tmp_path: pathlib.Path, cli_runner: CliRunner):
        tmp_gguf = tmp_path / "model.gguf"
        with open(tmp_gguf, "wb") as gguf_file:
            gguf_file.write(struct.pack("<I", GGUF_MAGIC))
        result = cli_runner.invoke(
            lab.ilab,
            [
                "--config=DEFAULT",
                "model",
                "upload",
                f"--model={tmp_gguf}",
                "--dest-type=hf",
                "--destination=testuser/testgguf",
                "--hf-token=foo",
            ],
        )
        assert (
            result.exit_code == 1
        ), f"command finished with an unexpected exit code. {result.stdout}"
        assert (
            f"Uploading GGUF model at {tmp_gguf} failed with the following Hugging Face Hub error:\n401 Client Error."
        ) in result.output

    def test_upload_oci(self, cli_runner: CliRunner):
        result = cli_runner.invoke(
            lab.ilab,
            [
                "--config=DEFAULT",
                "model",
                "upload",
                "--model=foo",
                "--dest-type=oci",
                "--destination=testuser/testgguf",
            ],
        )
        assert (
            result.exit_code == 1
        ), f"command finished with an unexpected exit code. {result.stdout}"
        assert result.output == "Uploading of type oci is not yet supported\n"

    def test_upload_s3(self, cli_runner: CliRunner):
        result = cli_runner.invoke(
            lab.ilab,
            [
                "--config=DEFAULT",
                "model",
                "upload",
                "--model=foo",
                "--dest-type=s3",
                "--destination=testuser/testgguf",
            ],
        )
        assert (
            result.exit_code == 1
        ), f"command finished with an unexpected exit code. {result.stdout}"
        assert result.output == "Uploading of type s3 is not yet supported\n"
