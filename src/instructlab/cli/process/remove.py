# SPDX-License-Identifier: Apache-2.0

# Third Party
import click

# First Party
from instructlab import clickext
from instructlab.process.process import (
    display_processes,
    filter_process_with_conditions,
    remove_process,
)


@click.command(name="remove")
@click.argument("process_uuid", required=True)
@click.option(
    "-f",
    "--force",
    is_flag=True,
    help="Used to forcefully remove processes. Use with caution as it permanently deletes process without further confirmation.",
)
@clickext.display_params
def remove(process_uuid: str, force: bool):
    """Remove process with uuid from process list."""

    process_to_remove = filter_process_with_conditions(process_uuid=process_uuid)
    if not process_to_remove:
        click.echo("No matching process found.")
        return

    click.echo("Process to be removed:")
    display_processes(process_to_remove)
    if force or click.confirm(
        "Are you sure you want to remove the process?",
        default=False,
    ):
        for remove_uuid in process_to_remove:
            remove_process(remove_uuid)
    else:
        click.secho("Aborted deletion.", fg="yellow")
