# SPDX-License-Identifier: Apache-2.0

# Standard
from unittest.mock import MagicMock, patch

# Third Party
from click.testing import CliRunner
from huggingface_hub.utils import HfHubHTTPError

# First Party
from instructlab import lab


class TestLabDownload:
    # When using `from X import Y` you need to understand that Y becomes part
    # of your module, so you should use `my_module.Y`` to patch.
    # When using `import X`, you should use `X.Y` to patch.
    # https://docs.python.org/3/library/unittest.mock.html#where-to-patch?
    @patch("instructlab.model.download.get_model_size_in_gb", return_value=5.00)
    @patch("instructlab.model.download.get_available_space_in_gb", return_value=10.00)
    @patch("instructlab.model.download.hf_hub_download")
    def test_download(
        self,
        mock_hf_hub_download,
        mock_available_space,
        mock_model_size,
        cli_runner: CliRunner,
    ):
        result = cli_runner.invoke(
            lab.ilab,
            [
                "--config=DEFAULT",
                "model",
                "download",
                "--hf-token=foo",
            ],
        )
        assert (
            result.exit_code == 0
        ), f"command finished with an unexpected exit code. {result.stdout}"
        assert mock_hf_hub_download.call_count == 3
        assert mock_available_space.return_value == 10.00
        assert mock_model_size.return_value == 5.00

    @patch("instructlab.model.download.get_model_size_in_gb", return_value=5.00)
    @patch("instructlab.model.download.get_available_space_in_gb", return_value=10.00)
    @patch(
        "instructlab.model.download.hf_hub_download",
        MagicMock(side_effect=HfHubHTTPError("Could not reach hugging face server")),
    )
    def test_download_error(
        self, mock_available_space, mock_model_size, cli_runner: CliRunner
    ):
        result = cli_runner.invoke(
            lab.ilab,
            [
                "--config=DEFAULT",
                "model",
                "download",
            ],
        )
        assert result.exit_code == 1, "command finished with an unexpected exit code"
        assert "Could not reach hugging face server" in result.output
        mock_available_space.assert_called_once()
        mock_model_size.assert_called_once()

    @patch("instructlab.model.download.get_oci_image_size_in_gb", return_value=5.00)
    @patch("instructlab.model.download.get_available_space_in_gb", return_value=10.00)
    @patch("instructlab.model.download.OCIDownloader.download")
    def test_oci_download(
        self,
        mock_oci_download,
        mock_available_space,
        mock_model_size,
        cli_runner: CliRunner,
    ):
        result = cli_runner.invoke(
            lab.ilab,
            [
                "--config=DEFAULT",
                "model",
                "download",
                "--repository=docker://quay.io/ai-lab/models/granite-7b-lab",
                "--release=latest",
            ],
        )
        assert result.exit_code == 0
        mock_oci_download.assert_called_once()
        assert mock_available_space.return_value == 10.00
        assert mock_model_size.return_value == 5.00

    def test_oci_download_repository_error(self, cli_runner: CliRunner):
        result = cli_runner.invoke(
            lab.ilab,
            [
                "--config=DEFAULT",
                "model",
                "download",
                "--repository=docker://quay.io/ai-lab/models/granite-7b-lab:latest",
            ],
        )
        assert result.exit_code == 1
        assert (
            "\nInvalid repository supplied:\n    Please specify tag/version 'latest' via --release"
            in result.output
        )

    @patch("instructlab.model.download.get_model_size_in_gb", return_value=15.00)
    @patch("instructlab.model.download.get_available_space_in_gb", return_value=10.00)
    def test_download_insufficient_space(
        self,
        mock_available_space,
        mock_model_size,
        cli_runner: CliRunner,
    ):
        result = cli_runner.invoke(
            lab.ilab,
            [
                "--config=DEFAULT",
                "model",
                "download",
                "--hf-token=foo",
            ],
        )
        assert result.exit_code == 1, f"Unexpected exit code: {result.exit_code}"
        assert "Model size to download:  15.00 GB" in result.output
        assert "Available local storage: 10.00 GB" in result.output
        assert "Insufficient space to download the model!" in result.output
        assert mock_available_space.return_value == 10.00
        assert mock_model_size.return_value == 15.00
