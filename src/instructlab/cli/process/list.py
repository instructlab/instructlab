# SPDX-License-Identifier: Apache-2.0

# Third Party
import click

# First Party
from instructlab import clickext
from instructlab.defaults import ILAB_PROCESS_STATUS
from instructlab.process.process import (
    display_processes,
    filter_process_with_conditions,
)


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
    uuid_list = filter_process_with_conditions(state=state)
    if not uuid_list:
        click.echo("No matching process found.")
        return
    display_processes(uuid_list)
