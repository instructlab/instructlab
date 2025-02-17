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
    process_registry["11111111-799a-42a7-af02-d0b68fddf19d"] = {
        "pid": 20001,
        "children_pids": [111, 222, 333],
        "type": "Generation",
        "log_file": "/Users/test/.local/share/instructlab/logs/generation/generation-11111111-799a-42a7-af02-d0b68fddf19d.log",
        "start_time": "2025-02-02T21:46:13",
        "status": ILAB_PROCESS_STATUS.RUNNING.value,
    }
    process_registry["22222222-799a-42a7-af02-d0b68fddf19d"] = {
        "pid": 20002,
        "children_pids": [111, 222, 333],
        "type": "Generation",
        "log_file": "/Users/test/.local/share/instructlab/logs/generation/generation-22222222-799a-42a7-af02-d0b68fddf19d.log",
        "start_time": "2025-02-02T21:46:13",
        "status": ILAB_PROCESS_STATUS.RUNNING.value,
    }
    process_registry["55555555-799a-42a7-af02-d0b68fddf19d"] = {
        "pid": 20005,
        "children_pids": [777, 888, 999],
        "type": "Generation",
        "log_file": "/Users/test/.local/share/instructlab/logs/generation/generation-55555555-799a-42a7-af02-d0b68fddf19d.log",
        "start_time": "2025-01-01T21:46:13",
        "status": ILAB_PROCESS_STATUS.ERRORED.value,
    }
    process_registry["66666666-799a-42a7-af02-d0b68fddf19d"] = {
        "pid": 20006,
        "children_pids": [777, 888, 999],
        "type": "Generation",
        "log_file": "/Users/test/.local/share/instructlab/logs/generation/generation-66666666-799a-42a7-af02-d0b68fddf19d.log",
        "start_time": "2025-01-11T21:46:13",
        "status": ILAB_PROCESS_STATUS.RUNNING.value,
    }
    os.makedirs(exist_ok=True, name=DEFAULTS.INTERNAL_DIR)
    with open(DEFAULTS.PROCESS_REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(process_registry, f)

    # Test single uuid with --force
    result = cli_runner.invoke(
        lab.ilab,
        [
            "--config=DEFAULT",
            "process",
            "remove",
            "11111111-799a-42a7-af02-d0b68fddf19d",
            "--force",
        ],
    )

    assert result.exit_code == 0
    assert (
        "Process with UUID 11111111-799a-42a7-af02-d0b68fddf19d removed"
        in result.output
    )

    # Test without --force and print the detial removal message
    result = cli_runner.invoke(
        lab.ilab,
        [
            "--config=DEFAULT",
            "process",
            "remove",
            "22222222-799a-42a7-af02-d0b68fddf19d",
        ],
        input="yes\n",
    )

    assert result.exit_code == 0
    assert "Generation" in result.output
    assert "20002" in result.output
    assert "22222222-799a-42a7-af02-d0b68fddf19d" in result.output
    assert (
        "/Users/test/.local/share/instructlab/logs/generation/generation-22222222-799a-42a7-af02-d0b68fddf19d.log"
        in result.output
    )
    assert "Running" in result.output
    assert "Are you sure you want to remove the process?" in result.output

    # Test with --state
    result = cli_runner.invoke(
        lab.ilab,
        [
            "--config=DEFAULT",
            "process",
            "prune",
            "--state",
            "errored",
            "--force",
        ],
    )

    assert result.exit_code == 0
    assert (
        "Process with UUID 55555555-799a-42a7-af02-d0b68fddf19d removed"
        in result.output
    )

    # Test with --older
    result = cli_runner.invoke(
        lab.ilab,
        ["--config=DEFAULT", "process", "prune", "--older", "1", "-f"],
    )

    assert result.exit_code == 0
    assert (
        "Process with UUID 66666666-799a-42a7-af02-d0b68fddf19d removed"
        in result.output
    )

    # Test no process
    result = cli_runner.invoke(
        lab.ilab,
        [
            "--config=DEFAULT",
            "process",
            "prune",
            "--state",
            "done",
            "--force",
        ],
    )

    assert result.exit_code == 0
    assert "No matching process found" in result.output
