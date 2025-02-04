# SPDX-License-Identifier: Apache-2.0

# Third Party
import click

# First Party
from instructlab import clickext
from instructlab.process.process import (
    filter_process_record_with_conditions,
    remove_process_record,
)


@click.command(name="remove")
@click.argument("input_uuid", required=False)
@click.option(
    "--older",
    type=int,
    help="Remove processes older than the specified number of days.",
)
@click.option(
    "--state",
    type=click.Choice(["Done", "Running", "Errored"], case_sensitive=False),
    help="Remove processes with the specified state.",
)
@click.option(
    "-f",
    "--force",
    is_flag=True,
    help="Used to forcefully remove process record. Use with caution as it permanently deletes the record from the file without further confirmation.",
)
@clickext.display_params
def remove(input_uuid: str, older: int, state: str, force: bool):
    """Remove process record from ilab proecess list."""
    if not input_uuid and not older and not state:
        click.secho("Please specify either a UUID, --older, or --state.", fg="red")
        raise click.exceptions.Exit(1)
    process_records_to_remove = filter_process_record_with_conditions(
        input_uuid=input_uuid, older=older, state=state
    )
    if force or click.confirm(
        f"Are you sure you want to remove process record?\nThe remove list: {process_records_to_remove}",
        default=False,
    ):
        if len(process_records_to_remove) > 0:
            for record_uuid in process_records_to_remove:
                remove_process_record(record_uuid)
        click.echo(f"Removed {len(process_records_to_remove)} process records.")
    else:
        click.secho("Aborted deletion.", fg="yellow")
