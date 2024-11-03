# Standard
import pathlib
import re
import textwrap
import time

# Third Party
from click.testing import CliRunner

# First Party
from instructlab import lab
from instructlab.cli.data.list import extract_model_and_timestamp


def extract_data_entries(table_output: str) -> str:
    lines = table_output.splitlines()
    data_entries = []
    for line in lines:
        if (
            line.startswith("|")
            and not line.startswith("| Dataset")
            and not line.startswith("+")
        ):
            clean_line = re.sub(r"\s+", " ", line.strip())
            data_entries.append(clean_line)
    return "\n".join(data_entries)


def test_data_list_command(cli_runner: CliRunner, tmp_path: pathlib.Path):
    test_dir = tmp_path / "list_datasets_test_dir"
    test_dir.mkdir()

    files_to_create = [
        "test_modelA_2024-10-27T08_34_48.jsonl",
        "train_modelA_2024-10-27T08_34_48.jsonl",
        "messages_modelB_2024-10-26T21_32_32.jsonl",
        "skills_train_msgs_2024-10-27T08_34_48.jsonl",
    ]

    file_info = []
    for filename in files_to_create:
        file_path = test_dir / filename
        file_path.write_text("test content")
        stat = file_path.stat()
        created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_ctime))
        formatted_size = f"{12.00:.2f} B"
        file_info.append((filename, created_at, formatted_size))

    node_datasets_dir = test_dir / "node_datasets_2024-10-27T08_34_48"
    node_datasets_dir.mkdir()
    node_file = node_datasets_dir / "compositional_skills_valid.jsonl"
    node_file.write_text("test content")
    node_stat = node_file.stat()
    node_created_at = time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(node_stat.st_ctime)
    )
    node_formatted_size = f"{12.00:.2f} B"

    result = cli_runner.invoke(
        lab.ilab,
        [
            "--config=DEFAULT",
            "data",
            "list",
            "--dataset-dirs",
            str(test_dir),
        ],
    )

    expected_output = textwrap.dedent(f"""
        modelA 2024-10-27 08:34:48
        +--------------------------------------------------------------------+---------------------+-----------+
        | Dataset                                                            | Created At          | File size |
        +--------------------------------------------------------------------+---------------------+-----------+
        | node_datasets_2024-10-27T08_34_48/compositional_skills_valid.jsonl | {node_created_at} | {node_formatted_size} |
        | skills_train_msgs_2024-10-27T08_34_48.jsonl                        | {file_info[3][1]} | {file_info[3][2]} |
        | test_modelA_2024-10-27T08_34_48.jsonl                              | {file_info[0][1]} | {file_info[0][2]} |
        | train_modelA_2024-10-27T08_34_48.jsonl                             | {file_info[1][1]} | {file_info[1][2]} |
        +--------------------------------------------------------------------+---------------------+-----------+

        modelB 2024-10-26 21:32:32
        +-------------------------------------------+---------------------+-----------+
        | Dataset                                   | Created At          | File size |
        +-------------------------------------------+---------------------+-----------+
        | messages_modelB_2024-10-26T21_32_32.jsonl | {file_info[2][1]} | {file_info[2][2]} |
        +-------------------------------------------+---------------------+-----------+
    """).strip()

    assert result.exit_code == 0
    assert extract_data_entries(result.output) == extract_data_entries(expected_output)


def test_extract_model_and_timestamp_model_file():
    filename = "train_modelA_2024-10-27T08_34_48.jsonl"
    group_key, is_general, timestamp = extract_model_and_timestamp(filename)

    assert group_key == "modelA 2024-10-27 08:34:48"
    assert is_general is False
    assert timestamp == "2024-10-27 08:34:48"


def test_extract_model_and_timestamp_general_file():
    filename = "skills_train_msgs_2024-10-27T08_34_48.jsonl"
    group_key, is_general, timestamp = extract_model_and_timestamp(filename)

    assert group_key == "General 2024-10-27 08:34:48"
    assert is_general is True
    assert timestamp == "2024-10-27 08:34:48"
