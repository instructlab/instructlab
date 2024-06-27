# Standard
import logging
import os

# Third Party
from click_didyoumean import DYMGroup
import click

# First Party
from instructlab import utils
from instructlab.configuration import ILAB_CONFIG_HOME, ILAB_DATA_HOME

logger = logging.getLogger(__name__)


@click.group(cls=DYMGroup)
@click.pass_context
def debug(ctx):
    """Command Group for debug-related ilab calls"""
    ctx.obj = ctx.parent.obj
    ctx.default_map = ctx.parent.default_map


@click.command()
@click.pass_context
@utils.display_params
def reset(
    ctx,
):
    """Start a local server"""

    # recursively removes the data at the config and share directories
    print(f"removing {ILAB_CONFIG_HOME}")
    os.system(f"rm -rf {ILAB_CONFIG_HOME}")
    print(f"removing {ILAB_DATA_HOME}")
    os.system(f"rm -rf {ILAB_DATA_HOME}")


debug.add_command(reset)
