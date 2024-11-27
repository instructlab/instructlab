# SPDX-License-Identifier: Apache-2.0

# Standard
from datetime import datetime
from typing import Callable
import json
import logging
import os
import subprocess
import sys
import time
import uuid

# Third Party
from filelock import FileLock
import psutil

# First Party
from instructlab.configuration import DEFAULTS
from instructlab.defaults import ILAB_PROCESS_MODES, ILAB_PROCESS_TYPES

logger = logging.getLogger(__name__)


class ProcessRegistry:
    def __init__(self):
        self.processes = {}

    def add_process(self, local_uuid, pid, children_pids, type, log_file, start_time):
        self.processes[str(local_uuid)] = {
            "pid": pid,
            "children_pids": children_pids,
            "type": type,
            "log_file": log_file,
            "start_time": datetime.strptime(
                start_time, "%Y-%m-%d %H:%M:%S"
            ).isoformat(),
        }

    def load_entry(self, key, value):
        self.processes[key] = value


def load_registry() -> ProcessRegistry:
    process_registry = ProcessRegistry()
    lock_path = DEFAULTS.PROCESS_REGISTRY_LOCK_FILE
    lock = FileLock(lock_path, timeout=1)
    """Load the process registry from a file, if it exists."""
    # we do not want a persistent registry in memory. This causes issues when in scenarios where you switch registry files (ex, in a unit test, or with multiple users)
    # but the registry with incorrect processes still exists in memory.
    with lock:
        if os.path.exists(DEFAULTS.PROCESS_REGISTRY_FILE):
            with open(DEFAULTS.PROCESS_REGISTRY_FILE, "r") as f:
                data = json.load(f)
                for key, value in data.items():
                    process_registry.load_entry(key=key, value=value)
        else:
            logger.debug("No existing process registry found. Starting fresh.")
    return process_registry


def save_registry(process_registry):
    """Save the current process registry to a file."""
    lock_path = DEFAULTS.PROCESS_REGISTRY_LOCK_FILE
    lock = FileLock(lock_path, timeout=1)
    with lock, open(DEFAULTS.PROCESS_REGISTRY_FILE, "w") as f:
        json.dump(dict(process_registry.processes), f)


class Tee:
    def __init__(self, log_file):
        """
        Initialize a Tee object.

        Args:
            log_file (str): Path to the log file where the output should be written.
        """
        self.log_file = log_file
        self.terminal = sys.stdout
        self.log = log_file  # Line-buffered

    def write(self, message):
        """
        Write the message to both the terminal and the log file.

        Args:
            message (str): The message to write.
        """
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        """
        Ensure all data is written to the terminal and the log file.
        """
        self.terminal.flush()
        self.log.flush()

    def close(self):
        """
        Close the log file.
        """
        if self.log:
            self.log.close()


def format_command(
    target: Callable, extra_imports: list[tuple[str, ...]], **kwargs
) -> str:
    """
    Formats a command given the target and any extra python imports to add

    Args:
        target: Callable
        extra_imports: list[tuple[str, ...]]
    Returns:
        cmd: str
    """
    # Prepare the subprocess command string
    cmd = (
        f"import {target.__module__}; {target.__module__}.{target.__name__}(**{kwargs})"
    )

    # Handle extra imports (if any)
    if extra_imports:
        import_statements = "\n".join(
            [f"from {imp[0]} import {', '.join(imp[1:])}" for imp in extra_imports]
        )
        cmd = f"{import_statements}\n{cmd}"
    return cmd


def start_process(cmd: str, log) -> tuple[int | None, list[int] | None]:
    """
    Starts a subprocess and captures PID and Children PIDs

    Args:
        cmd: str
        log: _FILE

    Returns:
        pid: int
        children_pids: list[int]
    """
    children_pids = []
    p = subprocess.Popen(
        ["python", "-c", cmd],
        universal_newlines=True,
        text=True,
        stdout=log,
        stderr=log,
        start_new_session=True,
        encoding="utf-8",
        bufsize=1,  # Line-buffered for real-time output
    )
    time.sleep(1)
    # we need to get all of the children processes spawned
    # to be safe, we will need to try and kill all of the ones which still exist when the user wants us to
    # however, representing this to the user is difficult. So let's track the parent pid and associate the children with it in the registry
    max_retries = 5
    retry_interval = 0.5  # seconds
    parent = psutil.Process(p.pid)
    for _ in range(max_retries):
        children = parent.children(recursive=True)
        if children:
            for child in children:
                children_pids.append(child.pid)
            break
        time.sleep(retry_interval)
    else:
        logger.debug("No child processes detected. Tracking parent process.")
    # Check if subprocess was successfully started
    if p.poll() is not None:
        logger.warning(f"Process {p.pid} failed to start.")
        return None, None  # Process didn't start
    return p.pid, children_pids


def add_process(
    process_mode: str,
    process_type: ILAB_PROCESS_TYPES,
    target: Callable,
    extra_imports: list[tuple[str, ...]],
    **kwargs,
):
    """
    Start a detached process using subprocess.Popen, logging its output.

    Args:
        process_mode (str): Mode we are running in, Detached or Attached.
        process_type (str): Type of process, ex: Generation.
        target (func): The target function to kick off in the subprocess or to run in the foreground.
        extra_imports (list[tuple(str...)]): a list of the extra imports to splice into the python subprocess command.

    Returns:
        None
    """
    process_registry = load_registry()
    if target is None:
        return None, None

    local_uuid = uuid.uuid1()
    log_file = None

    log_dir = os.path.join(DEFAULTS.LOGS_DIR, process_type.lower())

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, f"{process_type.lower()}-{local_uuid}.log")
    pid: int | None = os.getpid()
    children_pids: list[int] | None = []
    start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if process_mode == ILAB_PROCESS_MODES.DETACHED:
        assert isinstance(log_file, str)
        cmd = format_command(target=target, extra_imports=extra_imports, **kwargs)
        # Open the subprocess in the background, redirecting stdout and stderr to the log file
        with open(log_file, "a+") as log:
            pid, children_pids = start_process(cmd=cmd, log=log)
            if pid is None or children_pids is None:
                # process didn't start
                return None
            assert isinstance(pid, int) and isinstance(children_pids, list)
    # Add the process info to the shared registry
    process_registry.add_process(
        local_uuid=local_uuid,
        pid=pid,
        children_pids=children_pids,
        type=process_type,
        log_file=log_file,
        start_time=start_time_str,
    )
    logger.info(
        f"Started subprocess with PID {pid}. Logs are being written to {log_file}."
    )
    save_registry(
        process_registry=process_registry
    )  # Persist registry after adding process
    if process_mode == ILAB_PROCESS_MODES.ATTACHED:
        with open(log_file, "a+") as log:
            sys.stdout = Tee(log)
            sys.stderr = sys.stdout
            try:
                target(**kwargs)  # Call the function
            finally:
                # Restore the original stdout and stderr after the function completes
                process_registry.processes.pop(str(local_uuid))
                save_registry(process_registry=process_registry)
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__
