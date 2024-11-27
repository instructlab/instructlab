# Third Party
import click

# First Party
from instructlab import clickext
from instructlab.process.process import list_processes
from instructlab.utils import print_table


@click.command(name="list")
@clickext.display_params
def list():
    process_list = list_processes()
    if process_list is not None:
        print_table(["Type", "PID", "UUID", "Log File", "Runtime"], process_list)
