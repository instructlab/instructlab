# Third Party
import click

# First Party
from instructlab import clickext
from instructlab.process.process import attach_process


@click.command(name="attach")
@click.option(
    "--uuid", type=click.STRING, required=True, help="PID of the process to attach to"
)
@clickext.display_params
def attach(uuid):
    attach_process(uuid=uuid)
