# SPDX-License-Identifier: Apache-2.0

# Third Party
import click

# First Party
from instructlab import clickext
from instructlab.defaults import ILAB_PROCESS_STATUS
from instructlab.process.process import filter_processes
from instructlab.utils import print_table


@click.command(name="list")
@click.option(
    "--state",
    type=click.Choice(
        [s.value for s in ILAB_PROCESS_STATUS],
        case_sensitive=False,
    ),
    default=None,
    show_default=False,
    help="Filter processes with the specified state.",
)
@clickext.display_params
def list_processes(state: str):
    """List processes"""
    process_list = filter_processes(state=state)
    if process_list is not None and len(process_list) > 0:
        print_table(
            ["Type", "PID", "UUID", "Log File", "Runtime", "Status"], process_list
        )
    else:
        click.secho(
            "No processes found in registry",
            fg="red",
        )
        raise click.exceptions.Exit(0)
