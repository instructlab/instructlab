# Standard
from datetime import datetime
from multiprocessing import Manager
import json
import logging
import os
import signal
import subprocess
import sys
import time
import uuid

# Third Party
import psutil

# First Party
from instructlab.configuration import DEFAULTS
from instructlab.defaults import ILAB_PROCESS_MODES

logger = logging.getLogger(__name__)

# Create a Manager and a shared dictionary
manager = Manager()
process_registry = manager.dict()


def load_registry():
    """Load the process registry from a file, if it exists."""
    if os.path.exists(DEFAULTS.REGISTRY_FILE):
        with open(DEFAULTS.REGISTRY_FILE, "r") as f:
            data = json.load(f)
            for key, value in data.items():
                process_registry[key] = value
    else:
        logger.info("No existing process registry found. Starting fresh.")


def save_registry():
    """Save the current process registry to a file."""
    with open(DEFAULTS.REGISTRY_FILE, "w") as f:
        json.dump(dict(process_registry), f)


# Automatically load the registry when the module is imported
load_registry()


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


def add_process(process_mode: str, process_type: str, target, extra_imports, **kwargs):
    """
    Start a detached process using subprocess.Popen, logging its output.

    Args:
        task_name (str): Name of the task.
        process_registry (dict): Shared process registry.

    Returns:
        int: PID of the started process.
    """
    if target is None:
        return None, None

    local_uuid = uuid.uuid4()
    log_file = None

    log_dir = os.path.join(DEFAULTS.LOGS_DIR, process_type.lower())

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, f"{process_type.lower()}-{local_uuid}.log")
    pid = os.getpid()
    children_pids = []
    start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if process_mode == ILAB_PROCESS_MODES.DETACHED:
        # Prepare the subprocess command string
        cmd = f"import {target.__module__}; {target.__module__}.{target.__name__}(**{kwargs})"

        # Handle extra imports (if any)
        if extra_imports:
            import_statements = "\n".join(
                [f"from {imp[0]} import {', '.join(imp[1:])}" for imp in extra_imports]
            )
            cmd = f"{import_statements}\n{cmd}"
        assert isinstance(log_file, str)
        # Open the subprocess in the background, redirecting stdout and stderr to the log file
        with open(log_file, "a+") as log:
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
                logger.warning("No child processes detected. Tracking parent process.")
            pid = p.pid
            # Check if subprocess was successfully started
            if p.poll() is not None:
                logger.warning(f"Process {p.pid} failed to start.")
                return None  # Process didn't start

    # Add the process info to the shared registry
    process_registry[str(local_uuid)] = {
        "pid": pid,
        "children_pids": children_pids,
        "type": process_type,
        "log_file": log_file,
        "start_time": datetime.strptime(
            start_time_str, "%Y-%m-%d %H:%M:%S"
        ).isoformat(),
    }
    logger.info(
        f"Started subprocess with PID {pid}. Logs are being written to {log_file}."
    )
    save_registry()  # Persist registry after adding process
    if process_mode == ILAB_PROCESS_MODES.ATTACHED:
        with open(log_file, "a+") as log:
            sys.stdout = Tee(log)
            sys.stderr = sys.stdout
            try:
                target(**kwargs)  # Call the function
            finally:
                # Restore the original stdout and stderr after the function completes
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__


def list_processes():
    if not process_registry:
        logger.info("No processes currently in the registry.")
        return

    list_of_processes = []
    for local_uuid, entry in process_registry.items():
        now = datetime.now()

        # Calculate runtime
        runtime = now - datetime.fromisoformat(entry.get("start_time"))
        # Convert timedelta to a human-readable string (HH:MM:SS)
        total_seconds = int(runtime.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        runtime_str = f"{hours:02}:{minutes:02}:{seconds:02}"
        list_of_processes.append(
            (
                entry.get("type"),
                entry.get("pid"),
                local_uuid,
                entry.get("log_file"),
                runtime_str,
            )
        )

    return list_of_processes


def attach_process(uuid):
    """
    Attach to a running process and display its output in real-time.

    Args:
        pid (int): PID of the process to attach to.
        process_registry (dict): Shared process registry.
    """
    if uuid not in process_registry:
        logger.warning("Process not found.")
        return

    process_info = process_registry[uuid]
    pid = process_info["pid"]
    children_pids = process_info["children_pids"]
    log_file = process_info["log_file"]

    if not os.path.exists(log_file):
        logger.warning(
            "Log file not found. The process may not have started logging yet."
        )
        return

    logger.info(
        f"Attaching to process {pid}. Press Ctrl+C to detach and Ctrl+D to kill."
    )
    try:
        with open(log_file, "a+") as log:
            log.seek(0, os.SEEK_END)  # Move to the end of the log file
            while True:
                line = log.readline()
                # Check for non-empty and non-whitespace-only lines
                if line.strip():
                    print(line.strip())
                else:
                    time.sleep(0.1)  # Wait briefly before trying again
    except KeyboardInterrupt:
        logger.info("\nDetaching from and killing process.")
        stop_process(
            uuid=uuid,
            pid=pid,
            children_pids=children_pids,
            process_registry=process_registry,
        )


def stop_process(uuid, pid, children_pids, process_registry):
    """
    Stop a running process.

    Args:
        pid (int): PID of the process to stop.
        process_registry (dict): Shared process registry.
    """
    # we should kill the parent process, and also children processes.
    all_processes = [pid] + children_pids
    for process in all_processes:
        try:
            os.kill(process, signal.SIGKILL)
            logger.info(f"Process {process} terminated.")
        except ProcessLookupError:
            logger.warning(f"Process {process} was not running.")
    process_registry.pop(uuid, None)
    save_registry()


def get_latest_process() -> str:
    last_key = list(process_registry.keys())[-1]
    assert isinstance(last_key, str)
    return last_key
