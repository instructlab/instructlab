# SPDX-License-Identifier: Apache-2.0
# pylint: disable=R0801

# Standard
from pathlib import Path
from unittest.mock import patch
import os

# Third Party
from click.testing import CliRunner
import pytest

# First Party
from instructlab import lab


@pytest.mark.usefixtures("mock_mlx_package")
class TestLabModelTest:
    """Test collection for `ilab model test` command."""

    @patch("instructlab.utils.is_macos_with_m_chip", return_value=False)
    @patch("instructlab.model.backends.backends.is_model_gguf", return_value=True)
    @patch("instructlab.model.linux_test.response", return_value="answer!")
    def test_model_test_linux(
        self,
        is_macos_with_m_chip_mock,
        is_model_gguf_mock,
        response_mock,
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
            assert is_macos_with_m_chip_mock.call_count == 2
            assert is_model_gguf_mock.call_count == 2
            assert response_mock.call_count == 1
            assert "question?" in result.output
            assert "answer!" in result.output
            assert result.exit_code == 0
