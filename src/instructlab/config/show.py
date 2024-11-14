# SPDX-License-Identifier: Apache-2.0

# Standard
import sys

# Third Party
from ruamel.yaml import YAMLError, ruamel
from ruamel.yaml.comments import CommentedMap
import click

# First Party
from instructlab import clickext, configuration
from instructlab.utils import get_nested_value

# Initialize ruamel.yaml
yaml = ruamel.yaml.YAML()
yaml.indent(mapping=2, sequence=4, offset=2)


def strip_comments(config: CommentedMap) -> None:
    """Recursively remove comments from a ruamel.yaml CommentedMap structure."""
    config.ca.items.clear()
    for value in config.values():
        if isinstance(value, CommentedMap):
            strip_comments(value)


@click.command()
@click.pass_context
@click.option(
    "--key",
    "-k",
    "key_path",
    default=None,
    help="Show only a specific section of the configuration, e.g., -k chat or -k chat.context",
)
@click.option(
    "--without-comments",
    "-wc",
    "without_comments",
    is_flag=True,
    default=False,
    help="Show the config without comments. Can be used in conjunction with -k.",
)
@clickext.display_params
def show(ctx: click.Context, key_path: str, without_comments: bool) -> None:
    """Displays the current config as YAML with an option to extract a specific key."""

    try:
        commented_map = configuration.config_to_commented_map(ctx.obj.config)
        result = (
            commented_map if not key_path else get_nested_value(commented_map, key_path)
        )

        if without_comments:
            if isinstance(result, CommentedMap):
                strip_comments(result)

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
