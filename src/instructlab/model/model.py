# SPDX-License-Identifier: Apache-2.0

# Third Party
from click_didyoumean import DYMGroup
import click

# First Party
from instructlab import config

# Local
from .chat import chat
from .convert import convert
from .download import download
from .serve import serve
from .test import test
from .train import train


@click.group(cls=DYMGroup)
@click.version_option(package_name="instructlab")
@click.pass_context
# pylint: disable=redefined-outer-name
def model(ctx):
    """Command Group for Interacting with the Models in InstructLab.

    If this is your first time running ilab, it's best to start with `ilab init` to create the environment.
    """
    ctx.obj = ctx.parent.obj
    ctx.default_map = ctx.parent.default_map


model.add_command(serve)
model.add_command(train)
model.add_command(convert)
model.add_command(chat)
model.add_command(test)
model.add_command(download)
