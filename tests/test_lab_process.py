# Standard
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch
import json
import signal

# Third Party
from click.testing import CliRunner

# First Party
from instructlab import lab

sample_tasks = {
    "1234": {
        "pid": "1234",
        "command": "cmd",
        "args": ["--test"],
        "log_file": "/tmp/log",
        "start_time": "2024-01-01 12:00:00",
    },
    "5678": {
        "pid": "5678",
        "command": "cmd",
        "args": ["--test2"],
        "log_file": "/tmp/log2",
        "start_time": "2024-01-01 12:01:00",
    },
}


def populate_tasks(task_file: Path, tasks: Dict[str, Dict[str, Any]]):
    with open(task_file, "w", encoding="utf-8") as f:
        json.dump(tasks, f)


@patch("psutil.pid_exists")
def test_process_list(
    mock_pid_exists: MagicMock, temp_task_file: Path, cli_runner: CliRunner
):  # pylint: disable=unused-argument, redefined-outer-name
    populate_tasks(temp_task_file, sample_tasks)
    mock_pid_exists.side_effect = lambda pid: str(pid) in sample_tasks
    result = cli_runner.invoke(
        lab.ilab,
        [
            "--config=DEFAULT",
            "process",
            "list",
        ],
    )
    assert result.exit_code == 0
    assert "PID: 1234 (Running)" in result.output
    assert "PID: 5678 (Running)" in result.output
    assert "Command:" in result.output
    assert "--test" in result.output


@patch("os.kill")
@patch("psutil.pid_exists", return_value=True)
def test_process_stop(
    mock_pid_exists: MagicMock,
    mock_kill: MagicMock,
    temp_task_file: Path,
    cli_runner: CliRunner,
):  # pylint: disable=redefined-outer-name
    populate_tasks(temp_task_file, sample_tasks)
    mock_pid_exists.side_effect = lambda pid: str(pid) in sample_tasks
    result = cli_runner.invoke(
        lab.ilab,
        [
            "--config=DEFAULT",
            "process",
            "stop",
            "1234",
        ],
    )
    assert "Task with PID 1234 has been terminated." in result.output
    mock_kill.assert_called_once_with(1234, signal.SIGTERM)


@patch("instructlab.cli.process.process_manager.TASKS_FILE")
def test_clean(temp_task_file: Path, cli_runner: CliRunner):  # pylint: disable=redefined-outer-name
    populate_tasks(temp_task_file, sample_tasks)
    result = cli_runner.invoke(
        lab.ilab,
        [
            "--config=DEFAULT",
            "process",
            "clean",
            "-a",
        ],
    )
    assert "All task records have been cleared." in result.output
