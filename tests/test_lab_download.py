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
    @patch("instructlab.model.download.hf_hub_download")
    def test_download(self, mock_hf_hub_download, cli_runner: CliRunner):
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
        assert mock_hf_hub_download.call_count == 2

    @patch(
        "instructlab.model.download.hf_hub_download",
        MagicMock(side_effect=HfHubHTTPError("Could not reach hugging face server")),
    )
    def test_download_error(self, cli_runner: CliRunner):
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

    @patch("instructlab.model.download.OCIDownloader.download")
    def test_oci_download(self, mock_oci_download, cli_runner: CliRunner):
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
