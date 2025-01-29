# SPDX-License-Identifier: Apache-2.0

# Standard

# Third Party
import click

# First Party
from instructlab import clickext


@click.group(
    cls=clickext.LazyEntryPointGroup,
    ep_group="instructlab.command.process",
)
@click.pass_context
def process(ctx):
    """Command group for interacting with the processes run by InstructLab.

    If this is your first time running ilab, it's best to start with `ilab config init` to create the environment.
    """
    ctx.obj = ctx.parent.obj
    ctx.obj.ensure_config(ctx)
    ctx.default_map = ctx.parent.default_map
