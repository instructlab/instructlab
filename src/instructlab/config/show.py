# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, Dict
import json
import os

# Third Party
from rich.console import Console
from rich.syntax import Syntax
from ruamel.yaml import YAMLError, compat, ruamel
from ruamel.yaml.comments import CommentedMap
import click

# First Party
from instructlab import clickext, configuration

# Initialize ruamel.yaml
yaml = ruamel.yaml.YAML()
yaml.indent(mapping=2, sequence=4, offset=2)


def get_nested_value(data_dict: Dict[str, Any], path: str) -> Any:
    """Retrieve nested value using dot-separated path."""
    keys = path.split(".")
    for key in keys:
        data_dict = data_dict[key]
    return data_dict


def remove_comments(data: Any) -> Any:
    """Recursively remove comments from a YAML CommentedMap."""
    if isinstance(data, CommentedMap):
        return {k: remove_comments(v) for k, v in data.items()}
    return data


def check_color_support(console: Console) -> bool:
    """Checks if color output is supported and provides a warning if not."""
    no_color = bool(os.getenv("NO_COLOR"))
    supports_color = bool(console.color_system and console.is_terminal and not no_color)

    if not supports_color:
        reason = []
        if no_color:
            reason.append("NO_COLOR environment variable is set")
        if not console.is_terminal:
            reason.append("not running in a terminal")
        if not console.color_system:
            reason.append("the terminal does not support color")

        click.secho(f"Color output disabled due to: {', '.join(reason)}.", fg="yellow")

    return supports_color


@click.command()
@click.pass_context
@click.option(
    "--color",
    "-c",
    is_flag=True,
    help="Enable colored output if terminal supports, NO_COLOR not set, and output to terminal.",
)
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
    help="Extract a specific key from the configuration, can be used with -j and -c.",
)
@clickext.display_params
def show(ctx: click.Context, color: bool, json_format: bool, key_path: str) -> None:
    """Displays the current config as YAML or JSON, with an option to extract specific key."""

    console = Console()
    supports_color = check_color_support(console) if color else False

    try:
        commented_map = configuration.config_to_commented_map(ctx.obj.config)
        result = (
            commented_map if not key_path else get_nested_value(commented_map, key_path)
        )

        # If both -k and -c are specified, remove comments to prevent color inconsistencies.
        if key_path and color:
            result = remove_comments(result)

        stream = compat.StringIO()
        yaml.dump(result, stream)
        output = stream.getvalue()

        if isinstance(result, (dict, list)):
            if json_format:
                json_output = json.dumps(result, indent=4)
                click.echo(json_output)
            elif color and supports_color:
                syntax = Syntax(output, "yaml", line_numbers=False)
                console.print(syntax, style="green")
            else:
                click.echo(output)
        else:
            click.echo(result)

    except (YAMLError, KeyError) as e:
        click.secho(f"Error loading config or key not found: {e}.", fg="red")
        ctx.exit(2)
