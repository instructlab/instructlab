# SPDX-License-Identifier: Apache-2.0

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


@click.command(name="remove")
@click.argument("process_uuid", required=False)
@click.option(
    "--older",
    type=click.INT,
    help="Remove processes older than the specified number of days. Use 0 to remove all.",
)
@click.option(
    "--state",
    type=click.Choice(
        [s.value for s in ILAB_PROCESS_STATUS],
        case_sensitive=False,
    ),
    default=None,
    show_default=False,
    help="Remove processes with the specified state.",
)
@click.option(
    "-f",
    "--force",
    is_flag=True,
    help="Used to forcefully remove process. Use with caution as it permanently deletes process without further confirmation.",
)
@clickext.display_params
def remove(process_uuid: str, older: int, state: str, force: bool):
    """Remove process record from ilab proecess list."""

    if older is not None and older < 0:
        click.secho("Error: --older must be 0 or a positive integer.", fg="red")
        raise click.exceptions.Exit(1)

    if not process_uuid and older is None and not state:
        click.secho("Please specify either a UUID, --older, or --state.", fg="red")
        raise click.exceptions.Exit(1)

    process_records_to_remove = filter_process_with_conditions(
        process_uuid=process_uuid, older=older, state=state
    )
    if not process_records_to_remove:
        click.echo("No matching process found.")
        return

    click.echo("Processes to be removed:")
    display_remove_list(process_records_to_remove)
    if force or click.confirm(
        "Are you sure you want to remove the processes?",
        default=False,
    ):
        for record_uuid in process_records_to_remove:
            remove_process(record_uuid)
        click.echo(f"Removed {len(process_records_to_remove)} processes.")
    else:
        click.secho("Aborted deletion.", fg="yellow")
        return
