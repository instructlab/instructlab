# SPDX-License-Identifier: Apache-2.0
# Standard
import pathlib

# Third Party
from pydantic import ValidationError
from ruamel.yaml import YAML
import click

# First Party
from instructlab import clickext, configuration
from instructlab.utils import get_nested_value

yaml = YAML()


@click.command()
@click.pass_context
@click.option(
    "--key",
    "-k",
    help="Specify the configuration key to set value, e.g. general.log_level DEBUG",
)
@click.argument("value", required=False)
@clickext.display_params
def edit(ctx, key, value):
    """Launch $EDITOR to edit the configuration file or set the value with a specific key."""

    config_file = pathlib.Path(ctx.obj.config_file)

    with config_file.open("r", encoding="utf-8") as file:
        config_data = yaml.load(file)

    if key and value:
        try:
            commented_map = configuration.config_to_commented_map(ctx.obj.config)
            cfg_data = configuration.get_default_config().model_dump()

            # Top-level key
            if "." not in key:
                if key not in commented_map:
                    raise KeyError(f"Top-level key '{key}' not found in configuration.")
                current_value = commented_map[key]
                cfg_data[key] = value
            else:
                # Nested key
                current_value = get_nested_value(commented_map, key)
                update_section = get_nested_value(
                    cfg_data, ".".join(key.split(".")[:-1])
                )
                update_section[key.split(".")[-1]] = value

            # Validate the configuration
            configuration.Config(**cfg_data)

            if "." not in key:
                config_data[key] = value
            else:
                config_section = config_data
                keys = key.split(".")
                for k in keys[:-1]:
                    config_section = config_section.setdefault(k, {})
                config_section[keys[-1]] = value

            with config_file.open("w", encoding="utf-8") as file:
                yaml.dump(config_data, file)

            click.secho("Configuration Updated Successfully!", fg="green")
            click.secho(f"   Key       : {key}", fg="yellow")
            click.secho(f"   Old value : {current_value}", fg="blue")
            click.secho(f"   New value : {value}", fg="green")

        except KeyError:
            click.secho(f"Key not found '{key}'.", fg="red")
            ctx.exit(2)
        except ValidationError as err:
            click.secho(f"{err}", fg="red")
            ctx.exit(2)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            click.secho(f"Error: {exc}", fg="red")
            ctx.exit(2)

    elif key and not value:
        click.secho("Error: please provide both a key and a value to set.", fg="red")
        ctx.exit(2)
    else:
        click.edit(filename=str(config_file))
