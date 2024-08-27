# SPDX-License-Identifier: Apache-2.0
# Third Party
import click
import yaml

# First Party
from instructlab import clickext


@click.command()
@click.pass_context
@clickext.display_params
def show(ctx: click.Context) -> None:
    """Displays the current config as YAML"""
    # TODO: make this use pretty colors like jq/yq
    try:
        config_yaml = yaml.safe_load(ctx.obj.config.model_dump_json())
    except yaml.YAMLError as e:
        click.secho(f"Error loading config as YAML: {e}", fg="red")
        ctx.exit(2)
    click.echo(yaml.dump(config_yaml, indent=4))
