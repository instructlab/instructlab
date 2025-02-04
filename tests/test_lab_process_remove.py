# Standard
import json
import os

# Third Party
from click.testing import CliRunner

# First Party
from instructlab import lab
from instructlab.defaults import DEFAULTS, ILAB_PROCESS_STATUS


def test_process_record_remove(cli_runner: CliRunner):
    process_registry = {}
    process_registry["60000001-799a-42a7-af02-d0b68fddf19d"] = {
        "pid": 20001,
        "children_pids": [111, 222, 333],
        "type": "Generation",
        "log_file": "/Users/test/.local/share/instructlab/logs/generation/generation-60000001-799a-42a7-af02-d0b68fddf19d.log",
        "start_time": "2025-02-02T21:46:13",
        "status": ILAB_PROCESS_STATUS.RUNNING,
    }
    process_registry["60000002-799a-42a7-af02-d0b68fddf19d"] = {
        "pid": 20002,
        "children_pids": [111, 222, 333],
        "type": "Generation",
        "log_file": "/Users/test/.local/share/instructlab/logs/generation/generation-60000002-799a-42a7-af02-d0b68fddf19d.log",
        "start_time": "2025-02-02T21:46:13",
        "status": ILAB_PROCESS_STATUS.RUNNING,
    }
    process_registry["60000003-799a-42a7-af02-d0b68fddf19d"] = {
        "pid": 20003,
        "children_pids": [444, 555, 666],
        "type": "Generation",
        "log_file": "/Users/test/.local/share/instructlab/logs/generation/generation-60000003-799a-42a7-af02-d0b68fddf19d.log",
        "start_time": "2025-01-02T21:46:13",
        "status": ILAB_PROCESS_STATUS.DONE,
    }
    process_registry["60000004-799a-42a7-af02-d0b68fddf19d"] = {
        "pid": 20004,
        "children_pids": [777, 888, 999],
        "type": "Generation",
        "log_file": "/Users/test/.local/share/instructlab/logs/generation/generation-60000004-799a-42a7-af02-d0b68fddf19d.log",
        "start_time": "2025-01-02T21:46:13",
        "status": ILAB_PROCESS_STATUS.DONE,
    }
    process_registry["60000005-799a-42a7-af02-d0b68fddf19d"] = {
        "pid": 20005,
        "children_pids": [777, 888, 999],
        "type": "Generation",
        "log_file": "/Users/test/.local/share/instructlab/logs/generation/generation-60000005-799a-42a7-af02-d0b68fddf19d.log",
        "start_time": "2025-01-01T21:46:13",
        "status": ILAB_PROCESS_STATUS.ERRORED,
    }
    process_registry["60000006-799a-42a7-af02-d0b68fddf19d"] = {
        "pid": 20006,
        "children_pids": [777, 888, 999],
        "type": "Generation",
        "log_file": "/Users/test/.local/share/instructlab/logs/generation/generation-60000006-799a-42a7-af02-d0b68fddf19d.log",
        "start_time": "2025-01-11T21:46:13",
        "status": ILAB_PROCESS_STATUS.RUNNING,
    }
    os.makedirs(exist_ok=True, name=DEFAULTS.INTERNAL_DIR)
    with open(DEFAULTS.PROCESS_REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(process_registry, f)

    result = cli_runner.invoke(
        lab.ilab,
        [
            "--config=DEFAULT",
            "process",
            "remove",
            "60000001-799a-42a7-af02-d0b68fddf19d",
            "--force",
        ],
    )

    assert result.exit_code == 0
    assert (
        "Process with UUID 60000001-799a-42a7-af02-d0b68fddf19d removed"
        in result.output
    )
    assert "Removed 1 process records" in result.output

    result = cli_runner.invoke(
        lab.ilab,
        [
            "--config=DEFAULT",
            "process",
            "remove",
            "60000002-799a-42a7-af02-d0b68fddf19d",
        ],
        input="yes\n",
    )

    assert result.exit_code == 0
    assert "Are you sure you want to remove process record?" in result.output
    assert (
        "The remove list: ['60000002-799a-42a7-af02-d0b68fddf19d'] [y/N]:"
        in result.output
    )

    result = cli_runner.invoke(
        lab.ilab,
        [
            "--config=DEFAULT",
            "process",
            "remove",
            "--state",
            "errored",
            "--force",
        ],
    )

    assert result.exit_code == 0
    assert (
        "Process with UUID 60000005-799a-42a7-af02-d0b68fddf19d removed"
        in result.output
    )

    result = cli_runner.invoke(
        lab.ilab,
        [
            "--config=DEFAULT",
            "process",
            "prune",
            "-f",
        ],
    )

    assert result.exit_code == 0
    assert (
        "Process with UUID 60000003-799a-42a7-af02-d0b68fddf19d removed"
        in result.output
    )
    assert (
        "Process with UUID 60000004-799a-42a7-af02-d0b68fddf19d removed"
        in result.output
    )

    result = cli_runner.invoke(
        lab.ilab,
        ["--config=DEFAULT", "process", "remove", "--older", "1", "-f"],
    )

    assert result.exit_code == 0
    assert (
        "Process with UUID 60000006-799a-42a7-af02-d0b68fddf19d removed"
        in result.output
    )
