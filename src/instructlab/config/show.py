# SPDX-License-Identifier: Apache-2.0

# Standard

# Third Party
from ruamel.yaml import YAMLError, ruamel, compat
import click
from rich.console import Console
from rich.syntax import Syntax

# First Party
from instructlab import clickext, configuration

# Initialize ruamel.yaml
yaml = ruamel.yaml.YAML()
yaml.indent(mapping=2, sequence=4, offset=2)


@click.command()
@click.pass_context
@click.option('--color', '-c', is_flag=True, help="Enable colored output.")
@clickext.display_params
def show(ctx: click.Context, color: bool) -> None:
    """Displays the current config as YAML"""
    # use pretty colors like jq/yq
    console = Console()
    try:
        commented_map = configuration.config_to_commented_map(ctx.obj.config)
        stream = compat.StringIO()
        yaml.dump(commented_map, stream)
        if color and console.color_system:
            syntax = Syntax(stream.getvalue(), "yaml", theme="monokai", line_numbers=True)
            console.print(syntax)
        else:
            print(stream.getvalue())
    except YAMLError as e:
        click.secho(f"Error loading config as YAML: {e}", fg="red")
        ctx.exit(2)
    except Exception as e:
        click.secho(f"An unexpected error occurred: {e}", fg="red")
        ctx.exit(1)