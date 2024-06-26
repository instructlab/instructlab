# SPDX-License-Identifier: Apache-2.0
# Standard
import logging

# Third Party
from click_didyoumean import DYMGroup
import click

# Local
from .sysinfo import sysinfo

logger = logging.getLogger(__name__)


@click.group(cls=DYMGroup)
@click.pass_context
def system(ctx):
    """Command group for all system-related command calls"""
    ctx.obj = ctx.parent.obj
    ctx.default_map = ctx.parent.default_map


system.add_command(sysinfo)
