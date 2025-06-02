# SPDX-License-Identifier: Apache-2.0

# Standard
from datetime import datetime, timedelta
from typing import Any, Callable, List, Optional
import json
import logging
import os
import pathlib
import signal
import subprocess
import sys
import time
import uuid

# Third Party
from filelock import FileLock
import psutil

# First Party
from instructlab.configuration import DEFAULTS
from instructlab.defaults import ILAB_PROCESS_MODES, ILAB_PROCESS_STATUS
from instructlab.utils import print_table

logger = logging.getLogger(__name__)


class Process:
    def __init__(
        self,
        pid: int,
        log_path: pathlib.Path,
        ptype: str,
        children: list[int] | None = None,
        start_time: datetime | None = None,
        status: str = ILAB_PROCESS_STATUS.RUNNING.value,
    ):
        self.pid = pid
        self.ptype = ptype
        self.log_path = log_path
        self.status = status

        self._children = children or []
        self._start_time: datetime = start_time or datetime.now()
        self._end_time: datetime | None = None

    @property
    def pids(self) -> list[int]:
        return [self.pid] + self._children

    def complete(self, status: str):
        self.status = status
        self._end_time = datetime.now()

    @property
    def completed(self) -> bool:
        return self.status in (
            ILAB_PROCESS_STATUS.DONE.value,
            ILAB_PROCESS_STATUS.ERRORED.value,
        )

    @property
    def started(self) -> bool:
        return self.log_path.exists()

    @property
    def runtime(self) -> timedelta:
        return (self._end_time or datetime.now()) - self._start_time

    @property
    def start_time(self) -> datetime:
        return self._start_time

    # We are making effort to retain the original process representation as was
    # used in the original implementation. Some fields are named differently
    # for this reason.
    def to_json(self) -> dict[str, Any]:
        res = {
            "pid": self.pid,
            "type": str(self.ptype),
            "log_file": str(self.log_path),
            "children_pids": self._children,
            "start_time": self._start_time.isoformat(),
            "status": str(self.status),
        }
        if self._end_time:
            res["end_time"] = self._end_time.isoformat()
        return res


ProcessMap = dict[str, Process]


class ProcessRegistry:
    def __init__(self, filepath: pathlib.Path | None = None):
        self._filepath = filepath or DEFAULTS.PROCESS_REGISTRY_FILE
        self._lock = FileLock(f"{filepath}.lock", timeout=1)
        self._processes: ProcessMap = {}

    @property
    def processes(self) -> ProcessMap:
        # Don't allow the caller to modify the internal registry state without
        # going through proper methods
        return self._processes.copy()

    def add(self, id_: str, process: Process) -> "ProcessRegistry":
        self._processes[id_] = process
        return self

    def remove(self, id_: str) -> Process | None:
        return self._processes.pop(id_, None)

    def load(self) -> "ProcessRegistry":
        if os.path.exists(self._filepath):
            with self._lock, open(self._filepath) as f:
                data = json.load(f)
                for key, value in data.items():
                    self.add(
                        key,
                        Process(
                            pid=value["pid"],
                            log_path=pathlib.Path(value["log_file"]),
                            ptype=value["type"],
                            children=value["children_pids"],
                            start_time=datetime.fromisoformat(value["start_time"]),
                            status=value["status"],
                        ),
                    )
        return self

    def persist(self) -> "ProcessRegistry":
        with self._lock, open(self._filepath, "w") as f:
            json.dump({id_: p.to_json() for id_, p in self._processes.items()}, f)
        return self


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
    process_type: str,
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
    process_registry = ProcessRegistry().load()
    if target is None:
        return None, None

    local_uuid = str(uuid.uuid1())
    log_file = None

    log_dir = os.path.join(DEFAULTS.LOGS_DIR, process_type.lower())

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, f"{process_type.lower()}-{local_uuid}.log")
    kwargs["log_file"] = log_file
    kwargs["local_uuid"] = local_uuid
    kwargs["process_mode"] = process_mode
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
    else:
        pid = os.getpid()
        children_pids = []

    # Add the process info to the shared registry
    process_registry.add(
        local_uuid,
        Process(
            pid=pid,
            log_path=pathlib.Path(log_file),
            ptype=process_type,
            children=children_pids,
        ),
    )
    logger.info(
        f"Started subprocess with PID {pid}. Logs are being written to {log_file}."
    )
    process_registry.persist()  # Persist registry after adding process
    if process_mode == ILAB_PROCESS_MODES.ATTACHED:
        with open(log_file, "a+") as log:
            sys.stdout = Tee(log)
            sys.stderr = sys.stdout
            try:
                target(**kwargs)  # Call the function
            finally:
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__


def all_processes_running(pids: list[int]) -> bool:
    """
    Returns if a process and all of its children are still running
    Args:
        pids (list): a list of all PIDs to check
    """
    return all(psutil.pid_exists(pid) for pid in pids)


def stop_process(local_uuid: str, remove=True):
    """
    Stop a running process.

    Args:
        local_uuid (str): uuid of the process to stop.
    """
    process_registry = ProcessRegistry().load()
    # we should kill the parent process, and also children processes.
    if local_uuid not in process_registry.processes:
        logger.warning("Process not found.")
        return

    process = process_registry.processes[local_uuid]
    for pid in process.pids:
        try:
            os.kill(pid, signal.SIGKILL)
            logger.info(f"Process {pid} terminated.")
        except (ProcessLookupError, PermissionError):
            logger.warning(f"Process {pid} was not running or could not be stopped.")

    if remove:
        process_registry.remove(local_uuid)
    else:
        # since we just killed the processes, we cannot depend on it to update itself, mark as done and set end time
        process.complete(ILAB_PROCESS_STATUS.DONE.value)
    process_registry.persist()


def complete_process(local_uuid: str, status):
    """
    Updates the status of a process.

    Args:
        local_uuid (str): uuid of the process to stop.
    """

    process_registry = ProcessRegistry().load()
    process = process_registry.processes.get(local_uuid, None)
    if process:
        process.complete(status)
    process_registry.persist()


def attach_process(local_uuid: str):
    """
    Attach to a running process and display its output in real-time.

    Args:
        local_uuid (str): UUID of the process to attach to
    """
    process_registry = ProcessRegistry().load()
    process = process_registry.processes.get(local_uuid, None)
    if not process:
        logger.warning("Process not found.")
        return

    if not process.started:
        logger.warning("The process may not have started yet.")
        return

    logger.info(f"Attaching to process {local_uuid}. Press Ctrl+C to detach and kill.")
    all_pids = process.pids
    if not all_processes_running(all_pids):
        return
    try:
        with open(process.log_path, "a+") as log:
            log.seek(0, os.SEEK_END)  # Move to the end of the log file
            while all_processes_running(all_pids):
                line = log.readline()
                # Check for non-empty and non-whitespace-only lines
                if line.strip():
                    print(line.strip())
                else:
                    time.sleep(0.1)  # Wait briefly before trying again
    except KeyboardInterrupt:
        logger.info("\nDetaching from and killing process.")
    finally:
        stop_process(local_uuid=local_uuid, remove=False)


def get_latest_process() -> str | None:
    """
    Returns the last process added to the registry to quickly allow users to attach to it.

    Returns:
        last_key (str): a string UUID to attach to
    """
    processes = ProcessRegistry().load().processes
    if not processes:
        return None
    return list(processes)[-1]


def remove_process(process_uuid: str):
    """Remove a process by its UUID and delete its log artifacts."""
    process_registry = ProcessRegistry().load()
    if process_uuid not in process_registry.processes:
        logger.info(f"Process with UUID {process_uuid} not found.")
        return

    process = process_registry.processes.get(process_uuid)
    log_file = process.log_path if process else None
    stop_process(process_uuid, remove=True)
    logger.debug(f"Stopped process {process_uuid}.")

    # Remove the log
    if log_file and os.path.exists(log_file):
        os.remove(log_file)
        logger.debug(f"Log file {log_file} removed.")

    logger.info(f"Process with UUID {process_uuid} removed.")


def filter_process_with_conditions(
    process_uuid: Optional[str] = None,
    state: Optional[str] = None,
    older: Optional[int] = None,
) -> List[str]:
    """Filter processes based on their uuid, state or older."""
    process_registry = ProcessRegistry().load()

    uuid_list = []
    for item_uuid, process in process_registry.processes.items():
        if process_uuid and item_uuid != process_uuid:
            continue

        if state and process.status != state:
            continue

        if older == 0:
            uuid_list.append(item_uuid)
            continue

        if older:
            start_time = process.start_time
            now = datetime.now()
            if (now - start_time).days < older:
                continue

        uuid_list.append(item_uuid)

    return uuid_list


def display_processes(uuid_list: List[str]):
    process_registry = ProcessRegistry().load()
    display_list = []

    for local_uuid, process in process_registry.processes.items():
        if local_uuid in uuid_list:
            hours, remainder = divmod(process.runtime.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            days, hours = divmod(hours, 24)
            runtime_str = f"{hours:02}:{minutes:02}:{seconds:02}"
            if days > 0:
                runtime_str = f"{days}d {runtime_str}"
            display_list.append(
                (
                    process.ptype,
                    process.pid,
                    local_uuid,
                    process.log_path,
                    runtime_str,
                    process.status,
                )
            )

    print_table(
        ["Type", "PID", "UUID", "Log File", "Runtime", "Status"],
        [tuple(map(str, row)) for row in display_list],
    )
