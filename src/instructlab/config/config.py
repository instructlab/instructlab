# SPDX-License-Identifier: Apache-2.0
# Third Party
import click

# First Party
from instructlab import clickext


@click.group(
    cls=clickext.LazyEntryPointGroup,
    ep_group="instructlab.command.config",
)
@click.pass_context
def config(ctx):
    """Command Group for Interacting with the Config of InstructLab.

    If this is your first time running ilab, it's best to start with `ilab config init` to create the environment.
    """
    ctx.obj = ctx.parent.obj
    if ctx.invoked_subcommand not in {"init"}:
        ctx.obj.ensure_config(ctx)
    ctx.default_map = ctx.parent.default_map
