# Standard
from unittest.mock import MagicMock, patch
import unittest

# Third Party
from click.testing import CliRunner
from huggingface_hub.utils import HfHubHTTPError

# First Party
from cli import lab

server_error = HfHubHTTPError("Could not reach hugging face server")


def mock_download_raise_exception(repo_id, revision, filename, local_dir):
    raise server_error


class TestLabDownload(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @patch("huggingface_hub.hf_hub_download")
    def test_download(self, _):
        runner = CliRunner()
        with runner.isolated_filesystem():
            lab.hf_hub_download = MagicMock()
            result = runner.invoke(lab.download)
            self.assertEqual(result.exit_code, 0)

    @patch("huggingface_hub.hf_hub_download")
    def test_download_error(self, _):
        runner = CliRunner()
        with runner.isolated_filesystem():
            lab.hf_hub_download = mock_download_raise_exception
            result = runner.invoke(lab.download)
            self.assertEqual(result.exit_code, 1)
            self.assertTrue(
                f"Downloading model failed with the following Hugging Face Hub error: {server_error}"
            )
