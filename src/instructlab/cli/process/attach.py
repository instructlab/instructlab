# Third Party
import click

# First Party
from instructlab import clickext


@click.command(name="attach")
@clickext.display_params
@click.option("--uuid", type=click.STRING, help="PID of the process to attach to")
@click.option("--latest", is_flag=True, help="attach to the latest process")
def attach(uuid, latest):
    # First Party
    from instructlab.process.process import attach_process, get_latest_process

    if latest:
        uuid = get_latest_process()
    attach_process(uuid=uuid)
