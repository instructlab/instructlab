# SPDX-License-Identifier: Apache-2.0

# Standard
from unittest.mock import MagicMock, patch, mock_open
import os
import io

# Third Party
from click.testing import CliRunner
from huggingface_hub.utils import HfHubHTTPError
from xdg_base_dirs import xdg_cache_home

# First Party
from instructlab import lab


class TestLabDownload:
    # When using `from X import Y` you need to understand that Y becomes part
    # of your module, so you should use `my_module.Y`` to patch.
    # When using `import X`, you should use `X.Y` to patch.
    # https://docs.python.org/3/library/unittest.mock.html#where-to-patch?
    # @patch("instructlab.model.download.dump_metadata")
    @patch("instructlab.model.download.hf_hub_download")  # Mocking hf_hub_download
    @patch('builtins.open', new_callable=mock_open())
    def test_download(self, mock_hf_hub_download, mock_open_file, cli_runner: CliRunner):
        # Simulating the CLI command execution
        result = cli_runner.invoke(
            lab.ilab,
            [
                "--config=DEFAULT",
                "model",
                "download",
            ],
        )

        # Print the result for debugging purposes
        print(result)

        # Asserting that the command ran successfully
        assert result.exit_code == 0, "command finished with an unexpected exit code"

        # Assert that hf_hub_download was called exactly once
        mock_hf_hub_download.assert_called_once()
        mock_open_file.assert_called_once_with('metadata.json', 'w')


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
