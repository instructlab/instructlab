# SPDX-License-Identifier: Apache-2.0
# Third Party
import click

# First Party
from instructlab import clickext


@click.command()
@click.pass_context
@clickext.display_params
def edit(
    ctx,
):
    """Launch $EDITOR to edit the configuration file."""
    click.edit(filename=ctx.obj.config_file)
