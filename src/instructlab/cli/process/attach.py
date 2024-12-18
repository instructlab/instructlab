# SPDX-License-Identifier: Apache-2.0

# Third Party
import click

# First Party
from instructlab import clickext
from instructlab.process.process import attach_process, get_latest_process


@click.command(name="attach")
@clickext.display_params
@click.option("--uuid", type=click.STRING, help="UUID of the process to attach to")
@click.option("--latest", is_flag=True, help="Attach to the latest process")
def attach(uuid, latest):
    if latest:
        uuid = get_latest_process()
        if uuid is None:
            click.secho(
                "No processes found in registry",
                fg="red",
            )
            raise click.exceptions.Exit(1)
    attach_process(local_uuid=uuid)
