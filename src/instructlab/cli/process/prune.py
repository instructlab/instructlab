# Third Party
import click

# First Party
from instructlab import clickext
from instructlab.defaults import ILAB_PROCESS_STATUS
from instructlab.process.process import (
    display_remove_list,
    filter_process_with_conditions,
    remove_process,
)


@click.command(name="prune")
@click.option(
    "--older",
    type=click.INT,
    default=7,
    help="Remove completed processes older than the specified number of days (defautl: 7 days). Use 0 to remove all.",
)
@click.option(
    "-f",
    "--force",
    is_flag=True,
    help="Used to forcefully remove process. Use with caution as it permanently deletes process without further confirmation.",
)
@clickext.display_params
def prune(older: int, force: bool):
    """Remove all completed processes older than a number of days."""

    if older is not None and older < 0:
        click.secho("Error: --older must be 0 or a positive integer.", fg="red")
        raise click.exceptions.Exit(1)

    process_records_to_remove = filter_process_with_conditions(
        older=older, state=ILAB_PROCESS_STATUS.DONE.value
    )

    if not process_records_to_remove:
        click.echo("No matching process found.")
        return

    click.echo("\nProcesses to be removed:\n")
    display_remove_list(process_records_to_remove)
    if force or click.confirm(
        "Are you sure you want to remove the proecesses?",
        default=False,
    ):
        for uuid in process_records_to_remove:
            remove_process(uuid)
        click.echo(f"Removed {len(process_records_to_remove)} processes.")
    else:
        click.secho("Aborted deletion.", fg="yellow")
        return
