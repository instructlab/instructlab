# Standard
from unittest.mock import patch
import pathlib
import re
import textwrap

# Third Party
from click.testing import CliRunner

# First Party
from instructlab import lab
from instructlab.utils import AnalyzeModelResult

MOCKED_MODELS = [
    AnalyzeModelResult(
        model_name="models/test_model_a",
        model_modification_time="2025-01-21 11:20:59",
        model_size="10.0 GB",
        model_path=pathlib.Path("/path/to/mocked/model_a"),
    ),
]


def extract_model_entries(table_output: str) -> str:
    lines = table_output.splitlines()
    data_entries = []
    for line in lines:
        if (
            line.startswith("|")
            and not line.startswith("| Model")
            and not line.startswith("+")
        ):
            clean_line = re.sub(r"\s+", " ", line.strip())
            data_entries.append(clean_line)
    return "\n".join(data_entries)


class TestModelList:
    @patch("instructlab.model.list.list_models", return_value=MOCKED_MODELS)
    def test_model_list_command(
        self, mock_list_models, cli_runner: CliRunner, tmp_path: pathlib.Path
    ):
        test_dir = tmp_path / "list_models_test_dir"
        test_dir.mkdir()

        result = cli_runner.invoke(
            lab.ilab,
            [
                "--config=DEFAULT",
                "model",
                "list",
                "--model-dirs",
                str(test_dir),
            ],
        )
        expected_output = textwrap.dedent("""
            +---------------------------+---------------------+---------+-------------------+
            | Model Name          | Last Modified       | Size    | Absolute path           |
            +---------------------------+---------------------+---------+-------------------+
            | models/test_model_a | 2025-01-21 11:20:59 | 10.0 GB | /path/to/mocked/model_a |
            +---------------------------+---------------------+---------+-------------------+
        """)
        mock_list_models.assert_called_once()
        assert result.exit_code == 0
        assert extract_model_entries(result.output) == extract_model_entries(
            expected_output
        )
