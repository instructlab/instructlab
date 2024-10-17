# SPDX-License-Identifier: Apache-2.0
# Standard
import sys

# Third Party
from ruamel.yaml import YAMLError, ruamel
import click

# First Party
from instructlab import clickext, configuration

# Initialize ruamel.yaml
yaml = ruamel.yaml.YAML()
yaml.indent(mapping=2, sequence=4, offset=2)


@click.command()
@click.pass_context
@clickext.display_params
def show(ctx: click.Context) -> None:
    """Displays the current config as YAML"""
    # TODO: make this use pretty colors like jq/yq
    try:
        commented_map = configuration.config_to_commented_map(ctx.obj.config)
    except YAMLError as e:
        click.secho(f"Error loading config as YAML: {e}", fg="red")
        ctx.exit(2)
    yaml.dump(commented_map, sys.stdout)
