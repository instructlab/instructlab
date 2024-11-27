# Standard
from datetime import datetime
from multiprocessing import Manager
import json
import logging
import os
import subprocess
import time
import uuid

# Third Party
import psutil

# First Party
from instructlab.configuration import DEFAULTS
from instructlab.defaults import ILAB_PROCESS_MODES, ILAB_PROCESS_TYPES

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


def add_process(
    process_mode: str, process_type: ILAB_PROCESS_TYPES, target, extra_imports, **kwargs
):
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

    if process_mode == ILAB_PROCESS_MODES.DETACHED:
        # root_logger = logging.getLogger()
        log_dir = os.path.join(DEFAULTS.LOGS_DIR, process_type.lower())

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_file = os.path.join(log_dir, f"{process_type.lower()}-{local_uuid}.log")
    # add_file_handler_to_logger(logger=root_logger, log_file=log_file)

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

    # Open the subprocess in the background, redirecting stdout and stderr to the log file
    with open(log_file, "a") as log:
        start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        p = subprocess.Popen(
            ["python", "-c", cmd],
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        time.sleep(1)
        # we need to get all of the children processes spawned
        # to be safe, we will need to try and kill all of the ones which still exist when the user wants us to
        # however, representing this to the user is difficult. So let's track the parent pid and associate the children with it in the registry
        max_retries = 5
        retry_interval = 0.5  # seconds
        parent = psutil.Process(p.pid)
        children_pids = []
        for _ in range(max_retries):
            children = parent.children(recursive=True)
            if children:
                for child in children:
                    children_pids.append(child.pid)
                break
            time.sleep(retry_interval)
        else:
            logger.warning("No child processes detected. Tracking parent process.")

        # Add the process info to the shared registry
        process_registry[str(local_uuid)] = {
            "pid": p.pid,
            "children_pids": children_pids,
            "type": process_type,
            "log_file": log_file,
            "start_time": datetime.strptime(
                start_time_str, "%Y-%m-%d %H:%M:%S"
            ).isoformat(),
        }
        # Check if subprocess was successfully started
        if p.poll() is not None:
            logger.warning(f"Process {p.pid} failed to start.")
            return None  # Process didn't start

        # Return the process ID and immediately exit the parent process
        pgid: int = os.getpgid(p.pid)
        logger.info(
            f"Started subprocess with PID {p.pid} and PGID {pgid}. Logs are being written to {log_file}."
        )
        save_registry()  # Persist registry after adding process
        return log_file, p.pid


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
