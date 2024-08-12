# SPDX-License-Identifier: Apache-2.0

# pylint: disable=too-many-lines

# Standard
import multiprocessing

# Third Party
from click_repl import repl  # type: ignore
import click

# First Party
from instructlab import clickext
from instructlab import configuration as cfg

# 'fork' is unsafe and incompatible with some hardware accelerators.
# Python 3.14 will switch to 'spawn' on all platforms.
multiprocessing.set_start_method(cfg.DEFAULTS.MULTIPROCESSING_START_METHOD, force=True)


@click.group(
    cls=clickext.ExpandAliasesGroup,
    ep_group="instructlab.command",
    alias_ep_group="instructlab.command.alias",
)
@click.option(
    "--config",
    "config_file",
    type=click.Path(),
    default=cfg.DEFAULTS.CONFIG_FILE,
    show_default=True,
    help="Path to a configuration file.",
)
@click.option(
    "-v",
    "--verbose",
    "debug_level",
    count=True,
    default=0,
    show_default=False,
    help="Enable debug logging (repeat for even more verbosity)",
)
@click.version_option(package_name="instructlab")
@click.pass_context
# pylint: disable=redefined-outer-name
def ilab(ctx, config_file, debug_level: int = 0):
    """CLI for interacting with InstructLab.

    If this is your first time running ilab, it's best to start with `ilab config init` to create the environment.
    """
    cfg.init(ctx, config_file, debug_level)


# Register the REPL with a custom prompt
def ilab_prompt():
    return "ilab # "


@ilab.command()
def shell():
    """Command Group for Interacting with the ilab REPL (Read-Eval-Print Loop) interface."""
    prompt_kwargs = {
        "message": ilab_prompt(),
    }
    repl(click.get_current_context(), prompt_kwargs=prompt_kwargs)
