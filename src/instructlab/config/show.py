# SPDX-License-Identifier: Apache-2.0

# Standard
import json

# Third Party
from rich.console import Console
from rich.syntax import Syntax
from ruamel.yaml import YAMLError, compat, ruamel
import click

# First Party
from instructlab import clickext, configuration

# Initialize ruamel.yaml
yaml = ruamel.yaml.YAML()
yaml.indent(mapping=2, sequence=4, offset=2)


def get_nested_value(data_dict, path):
    """Retrieve nested value using dot-separated path"""
    keys = path.split(".")
    for key in keys:
        data_dict = data_dict[key]
    return data_dict


@click.command()
@click.pass_context
@click.option("--color", "-c", is_flag=True, help="Enable colored output.")
@click.option(
    "--json",
    "-j",
    "json_format",
    is_flag=True,
    help="Output in JSON format instead of YAML.",
)
@click.option(
    "--key",
    "-k",
    "key_path",
    default=None,
    help="Extract a specific key from the configuration.",
)
@clickext.display_params
def show(ctx: click.Context, color: bool, json_format: bool, key_path: str) -> None:
    """Displays the current config as YAML or JSON, with an option to extract specific key."""
    console = Console()
    try:
        commented_map = configuration.config_to_commented_map(ctx.obj.config)
        result = (
            commented_map if not key_path else get_nested_value(commented_map, key_path)
        )

        stream = compat.StringIO()
        yaml.dump(result, stream)
        output = stream.getvalue()

        if isinstance(result, (dict, list)):
            if json_format:
                json_output = json.dumps(result, indent=4)
                print(json_output)
            elif color and console.color_system:
                syntax = Syntax(output, "yaml", theme="monokai", line_numbers=False)
                console.print(syntax)
            else:
                print(output)
        else:
            print(result)

    except (YAMLError, KeyError) as e:
        click.secho(f"Error loading config or key not found: {e}.", fg="red")
        ctx.exit(2)
