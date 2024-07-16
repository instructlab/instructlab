# SPDX-License-Identifier: Apache-2.0
# pylint: disable=R0801

# Standard
from pathlib import Path
from unittest.mock import patch
import os

# Third Party
from click.testing import CliRunner
import httpx
import pytest

# First Party
from instructlab import lab
from instructlab.model.backends.backends import BackendServer


@pytest.mark.usefixtures("mock_mlx_package")
class TestLabModelTest:
    """Test collection for `ilab model test` command."""

    class ServerMock(BackendServer):
        # py lint: disable=W0613
        def __init__(self):
            super().__init__("", "", "", "", "", 0)

        def run_detached(
            self, http_client: httpx.Client | None = None, background: bool = True
        ) -> str:
            return "api_base_mock"

        def run(self):
            pass

        def shutdown(self):
            pass

    @patch("instructlab.utils.is_macos_with_m_chip", return_value=False)
    @patch("instructlab.model.linux_test.response", return_value="answer!")
    @patch(
        "instructlab.model.backends.backends.select_backend", return_value=ServerMock()
    )
    def test_model_test_linux(
        self,
        select_backend_mock,
        response_mock,
        is_macos_with_m_chip_mock,
    ):
        runner = CliRunner()
        with runner.isolated_filesystem():
            os.mkdir("generated")
            Path("test_file.jsonl").write_text(
                '{"system": "", "user": "question?", "assistant": ""}', encoding="utf-8"
            )
            result = runner.invoke(
                lab.ilab,
                ["--config=DEFAULT", "model", "test", "--test_file", "test_file.jsonl"],
            )
            assert select_backend_mock.call_count
            assert response_mock.call_count
            assert is_macos_with_m_chip_mock.call_count
            assert "question?" in result.output
            assert "answer!" in result.output
            assert result.exit_code == 0
