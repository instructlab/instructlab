# SPDX-License-Identifier: Apache-2.0
import click
import yaml

from instructlab import utils


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
