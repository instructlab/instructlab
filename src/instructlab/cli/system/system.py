# SPDX-License-Identifier: Apache-2.0

# Standard
import logging

# Third Party
import click

# First Party
from instructlab import clickext

logger = logging.getLogger(__name__)


@click.group(
    cls=clickext.LazyEntryPointGroup,
    ep_group="instructlab.command.system",
)
@click.pass_context
def system(ctx):
    """Execute system commands."""
    ctx.obj = ctx.parent.obj
    ctx.default_map = ctx.parent.default_map
