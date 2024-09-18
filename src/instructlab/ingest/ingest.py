# SPDX-License-Identifier: Apache-2.0

# Standard

# Third Party
import click

# First Party
from instructlab import clickext


@click.group(
    cls=clickext.LazyEntryPointGroup,
    ep_group="instructlab.command.ingest",
)
@click.pass_context
def ingest(ctx):
    """Command Group for Interacting with the Knowledge Docs"""
    ctx.obj = ctx.parent.obj
    ctx.default_map = ctx.parent.default_map
