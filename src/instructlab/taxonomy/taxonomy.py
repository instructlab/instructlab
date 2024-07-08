# SPDX-License-Identifier: Apache-2.0

# Third Party
from click_didyoumean import DYMGroup
import click

# Local
from .diff import diff


@click.group(cls=DYMGroup)
@click.pass_context
def taxonomy(ctx):
    """Command Group for Interacting with the Taxonomy of InstructLab.

    If this is your first time running ilab, it's best to start with `ilab config init` to create the environment.
    """
    ctx.obj = ctx.parent.obj
    ctx.obj.ensure_config(ctx)
    ctx.default_map = ctx.parent.default_map


taxonomy.add_command(diff)
