# SPDX-License-Identifier: Apache-2.0

# This code instantiates the CLI ilab rag group.

# Standard

# Third Party
import click

# First Party
from instructlab import clickext


@click.group(cls=clickext.LazyEntryPointGroup, ep_group="instructlab.command.rag")
@click.pass_context
@clickext.display_params
def rag(ctx):
    """Command group for interacting with the RAG for InstructLab.

    If this is your first time running ilab, it's best to start with `ilab config init` to create the environment.
    """
    ctx.obj = ctx.parent.obj
    ctx.obj.ensure_config(ctx)
    ctx.default_map = ctx.parent.default_map
