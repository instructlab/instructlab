# SPDX-License-Identifier: Apache-2.0

# Standard
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union, cast
import json

# First Party
from instructlab.defaults import DEFAULTS

TASKS_FILE = (Path(DEFAULTS.PROCESS_DIR) / "process_tasks.json").expanduser()


def list_tasks() -> Dict[str, Dict[str, Union[str, List[str]]]]:
    """List all tasks from the tasks file."""
    if TASKS_FILE.exists():
        with open(TASKS_FILE, "r", encoding="utf-8") as f:
            return cast(Dict[str, Dict[str, Union[str, List[str]]]], json.load(f))
    return {}


def save_tasks(tasks: Dict[str, Dict[str, Union[str, List[str]]]]) -> None:
    """Save the tasks to the tasks file."""
    with open(TASKS_FILE, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2)


def register_task(pid: int, command: str, args: List[str], log_file: str) -> None:
    """Register a new task."""
    tasks = list_tasks()
    tasks[str(pid)] = {
        "pid": str(pid),
        "command": command,
        "args": args,
        "log_file": log_file,
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_tasks(tasks)


def remove_task(pid: int) -> None:
    """Remove a task by PID."""
    tasks = list_tasks()
    if str(pid) in tasks:
        del tasks[str(pid)]
        save_tasks(tasks)
