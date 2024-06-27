# SPDX-License-Identifier: Apache-2.0
# Third Party
from click_didyoumean import DYMGroup
import click
import yaml

# First Party
from instructlab import utils

# Local
from .init import init
from .show import show


@click.group(cls=DYMGroup)
@click.pass_context
def config(ctx):
    """Command Group for Interacting with the Config of InstructLab.

    If this is your first time running ilab, it's best to start with `ilab config init` to create the environment.
    """
    ctx.obj = ctx.parent.obj
    ctx.default_map = ctx.parent.default_map


config.add_command(init)
config.add_command(show)
