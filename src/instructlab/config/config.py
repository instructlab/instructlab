# SPDX-License-Identifier: Apache-2.0
import yaml

# Third Party
from click_didyoumean import DYMGroup
import click

from instructlab import utils

# Local
from .init import init


@click.group(cls=DYMGroup)
@click.pass_context
def config(ctx):
    """Command Group for Interacting with the Config of InstructLab.

    If this is your first time running ilab, it's best to start with `ilab config init` to create the environment.
    """
    ctx.obj = ctx.parent.obj
    ctx.default_map = ctx.parent.default_map


config.add_command(init)


@click.command()
@click.pass_context
@utils.display_params
def show(
    ctx,
):
    """Displays the current config as YAML"""
    # TODO: make this use pretty colors like jq/yq
    config_yaml = yaml.load(ctx.obj.config.model_dump_json(), Loader=yaml.FullLoader)
    print(yaml.dump(config_yaml, indent=4))


config.add_command(show)
