# Standard
import datetime
import json
import os
import textwrap

# Third Party
from click.testing import CliRunner

# First Party
from instructlab import lab
from instructlab.defaults import DEFAULTS


def extract_process_entries(table_output: str) -> str:
    lines = table_output.splitlines()
    entries = []

    for line in lines:
        if line.startswith("|") and not line.startswith("+"):
            # Split the line into columns based on the pipe delimiter
            columns = [col.strip() for col in line.split("|")[1:-1]]

            # Remove the "Runtime" column (last column)
            desired_columns = columns[:-1]

            # Clean and join the remaining columns
            clean_line = " | ".join(desired_columns)
            entries.append(clean_line)

    return "\n".join(entries)


def test_process_list(cli_runner: CliRunner):
    process_registry = {}
    start_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    process_registry["63866b91-799a-42a7-af02-d0b68fddf19d"] = {
        "pid": 26372,
        "children_pids": [111, 222, 333],
        "type": "Generation",
        "log_file": "/Users/charliedoern/.local/share/instructlab/logs/generation/generation-63866b91-799a-42a7-af02-d0b68fddf19d.log",
        "start_time": datetime.datetime.strptime(
            start_time_str, "%Y-%m-%d %H:%M:%S"
        ).isoformat(),
    }
    # create registry json, place it in the proper dir
    os.makedirs(exist_ok=True, name=DEFAULTS.INTERNAL_DIR)
    # os.remove(DEFAULTS.REGISTRY_FILE)
    print(DEFAULTS.REGISTRY_FILE)
    with open(DEFAULTS.REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(process_registry, f)
    # list processes and expect output
    result = cli_runner.invoke(
        lab.ilab,
        [
            "--config=DEFAULT",
            "process",
            "list",
        ],
    )

    expected_output = textwrap.dedent("""
+------------+-------+--------------------------------------+------------------------------------------------------------------------------------------------------------------+----------+
| Type       | PID   | UUID                                 | Log File                                                                                                         | Runtime  |
+------------+-------+--------------------------------------+------------------------------------------------------------------------------------------------------------------+----------+
| Generation | 26372 | 63866b91-799a-42a7-af02-d0b68fddf19d | /Users/charliedoern/.local/share/instructlab/logs/generation/generation-63866b91-799a-42a7-af02-d0b68fddf19d.log | 00:00:06 |
+------------+-------+--------------------------------------+------------------------------------------------------------------------------------------------------------------+----------+
    """).strip()

    assert result.exit_code == 0, result.output
    assert extract_process_entries(result.output) == extract_process_entries(
        expected_output
    ), result.output
