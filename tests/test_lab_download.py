# SPDX-License-Identifier: Apache-2.0

# Standard
from unittest.mock import MagicMock, patch
import unittest

# Third Party
from click.testing import CliRunner
from huggingface_hub.utils import HfHubHTTPError

# First Party
from cli import lab


class TestLabDownload(unittest.TestCase):
    # When using `from X import Y` you need to understand that Y becomes part
    # of your module, so you should use `my_module.Y`` to patch.
    # When using `import X`, you should use `X.Y` to patch.
    # https://docs.python.org/3/library/unittest.mock.html#where-to-patch?
    @patch("cli.lab.hf_hub_download")
    def test_download(self, mock_hf_hub_download):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(lab.download)
            self.assertEqual(
                result.exit_code, 0, "command finished with an unexpected exit code"
            )
            mock_hf_hub_download.assert_called_once()

    @patch(
        "cli.lab.hf_hub_download",
        MagicMock(side_effect=HfHubHTTPError("Could not reach hugging face server")),
    )
    def test_download_error(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(lab.download)
            self.assertEqual(
                result.exit_code, 1, "command finished with an unexpected exit code"
            )
            self.assertIn("Could not reach hugging face server", result.output)
