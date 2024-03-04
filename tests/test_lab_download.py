# Standard
from http.server import HTTPServer
import threading
import unittest

# Third Party
from click.testing import CliRunner

# First Party
from cli import lab

# Local
from .hf_mock_server import HuggingFaceTestServer


class TestLabDownload(unittest.TestCase):
    port = 8000
    host = "localhost"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def setUpClass(cls):
        cls.hf_server = HTTPServer((cls.host, cls.port), HuggingFaceTestServer)
        cls.server_thread = threading.Thread(target=cls.hf_server.serve_forever)
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.hf_server.shutdown()
        cls.server_thread.join()

    def test_lab_download_happy(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = CliRunner().invoke(lab.init, ["--interactive"])
            assert result.exit_code == 0
            self.assertEqual(result.exit_code, 0)
            result = CliRunner().invoke(
                lab.download, ["--hf-endpoint", f"http://{self.host}:{self.port}/"]
            )
            self.assertEqual(result.exit_code, 0)
