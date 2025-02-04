# Third Party
import click

# First Party
from instructlab import clickext
from instructlab.process.process import (
    filter_process_record_with_conditions,
    remove_process_record,
)


@click.command(name="prune")
@click.option(
    "--older",
    type=int,
    required=True,
    default=7,
    help="Remove completed processes older than the specified number of days (defautl: 7 days).",
)
@click.option(
    "-f",
    "--force",
    is_flag=True,
    help="Used to forcefully remove process record. Use with caution as it permanently deletes the record from the file without further confirmation.",
)
@clickext.display_params
def prune(older: int, force: bool):
    """Remove all completed process records older than a number of days."""
    process_records_to_remove = filter_process_record_with_conditions(
        older=older, state="done"
    )
    if force or click.confirm(
        f"Are you sure you want to remove process record?\nThe remove list: {process_records_to_remove}",
        default=False,
    ):
        if len(process_records_to_remove) > 0:
            for uuid in process_records_to_remove:
                remove_process_record(uuid)
        click.echo(f"Removed {len(process_records_to_remove)} process records.")
    else:
        click.secho("Aborted deletion.", fg="yellow")
