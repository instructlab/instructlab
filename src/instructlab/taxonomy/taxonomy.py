# SPDX-License-Identifier: Apache-2.0

# Third Party
from click_didyoumean import DYMGroup
import click

# First Party
from instructlab import config

# Local
from .diff import diff


@click.group(cls=DYMGroup)
@click.version_option(package_name="instructlab")
@click.pass_context
def taxonomy(ctx):
    """Command Group for Interacting with the Taxonomy of InstructLab.

    If this is your first time running ilab, it's best to start with `ilab init` to create the environment.
    """
    ctx.obj = ctx.parent.obj
    ctx.default_map = ctx.parent.default_map


taxonomy.add_command(diff)
