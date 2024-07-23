# SPDX-License-Identifier: Apache-2.0

# pylint: disable=redefined-builtin
# Third Party
import click

# First Party
from instructlab import clickext


@click.group(
    cls=clickext.LazyEntryPointGroup,
    ep_group="instructlab.command.model",
)
@click.pass_context
# pylint: disable=redefined-outer-name
def model(ctx):
    """Command Group for Interacting with the Models in InstructLab.

    If this is your first time running ilab, it's best to start with `ilab config init` to create the environment.
    """
    ctx.parent.obj.ensure_config(ctx)
    ctx.obj = ctx.parent.obj
    ctx.default_map = ctx.parent.default_map
