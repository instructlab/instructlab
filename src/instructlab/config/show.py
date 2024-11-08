# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, Dict
import json
import sys

# Third Party
from ruamel.yaml import YAMLError, ruamel
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


@click.command()
@click.pass_context
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
    help="Show only a specific section of the configuration. Can be used in conjunction with -j. e.g. -k chat or -k chat.context",
)
@clickext.display_params
def show(ctx: click.Context, json_format: bool, key_path: str) -> None:
    """Displays the current config as YAML or JSON, with an option to extract a specific key."""

    try:
        commented_map = configuration.config_to_commented_map(ctx.obj.config)
        result = (
            commented_map if not key_path else get_nested_value(commented_map, key_path)
        )

        if json_format:
            json_output = json.dumps(result, indent=4)
            click.echo(json_output)
        else:
            if isinstance(result, (dict, list)):
                yaml.dump(result, sys.stdout)
            else:
                click.echo(result)

    except KeyError:
        click.secho(f"Key not found: '{key_path}'", fg="red")
        ctx.exit(2)

    except YAMLError as e:
        click.secho(f"Error loading config: {e}.", fg="red")
        ctx.exit(2)
