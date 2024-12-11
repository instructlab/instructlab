# Standard
import os
import signal
import subprocess

# Third Party
import click
import psutil

# First Party
from instructlab import clickext
from instructlab.cli.process.process_manager import (
    TASKS_FILE,
    list_tasks,
    remove_task,
    save_tasks,
)
from instructlab.defaults import DEFAULT_INDENT


def is_task_running(pid: int) -> bool:
    """Check if a task with the given PID is still running."""
    try:
        return psutil.pid_exists(pid)
    except psutil.NoSuchProcess:
        return False


@click.command(name="list")
@clickext.display_params
def list_processes() -> None:
    """List all running background processes."""
    tasks = list_tasks()
    if not tasks:
        click.secho("No background processes found.", fg="yellow")
        return

    for pid_str, info in tasks.items():
        try:
            pid = int(pid_str)
            status = "Running" if psutil.pid_exists(pid) else "Stopped"

            # Format output
            click.secho(
                f"PID: {pid} ({status})", fg="green" if status == "Running" else "red"
            )
            click.echo(
                f"Command:\n{DEFAULT_INDENT}{info['command']} {' '.join(info['args'])}"
            )
            click.echo(f"Log File:\n{DEFAULT_INDENT}{info['log_file']}")
            click.echo(f"Start Time:\n{DEFAULT_INDENT}{info['start_time']}\n")
        except ValueError:
            click.secho(f"Invalid PID format: {pid_str}", fg="red")


@click.command(name="attach")
@clickext.display_params
@click.argument("pid", type=int)
def process_attach(pid: int) -> None:
    """Attach to a background process's log."""
    tasks = list_tasks()
    task = tasks.get(str(pid))
    if not task:
        click.secho(f"No task found with PID {pid}.", fg="red")
        return

    log_file = task["log_file"]
    click.echo(
        f"Attaching to a background process's log:\n{DEFAULT_INDENT}PID: {pid}\n{DEFAULT_INDENT}Log: {log_file}\n"
    )
    try:
        subprocess.run(["tail", "-f", str(log_file)], check=True)
    except subprocess.CalledProcessError as e:
        click.secho(
            f"\nFailed to run tail command:\n{DEFAULT_INDENT}{e}.\n{DEFAULT_INDENT}stderr: {e.stderr}",
            fg="red",
        )
        raise click.exceptions.Exit(1)


@click.command(name="stop")
@clickext.display_params
@click.argument("pid", type=int)
def process_stop(pid: int) -> None:
    """Stop a background process by PID, check status with `list` first."""
    tasks = list_tasks()
    task = tasks.get(str(pid))
    if not task:
        click.secho(f"No task found with PID {pid}.", fg="red")
        return

    try:
        os.kill(pid, signal.SIGTERM)
        remove_task(pid)
        click.secho(f"Task with PID {pid} has been terminated.", fg="green")
    except OSError as e:
        click.secho(f"Failed to terminate PID {pid}: {e}", fg="red")


@click.command(name="clean")
@clickext.display_params
@click.option("-a", "--all", "clean_all", is_flag=True, help="Clean all task records.")
@click.argument("pid", required=False, type=int)
def process_record_clean(clean_all: bool, pid: int) -> None:
    """Clean up task records."""
    if clean_all:
        TASKS_FILE.write_text("{}", encoding="utf-8")
        click.secho("All task records have been cleared.", fg="green")
    elif pid is not None:
        tasks = list_tasks()
        if str(pid) in tasks:
            del tasks[str(pid)]
            save_tasks(tasks)
            click.secho(f"Task with PID {pid} has been removed.", fg="green")
        else:
            click.secho(f"No task found with PID {pid}.", fg="red")
    else:
        click.secho("You must specify either -a/--all or a PID to clean.", fg="red")
