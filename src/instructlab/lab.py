# SPDX-License-Identifier: Apache-2.0

# pylint: disable=too-many-lines

# Standard
import multiprocessing

# Third Party
import click

# First Party
from instructlab import configuration as cfg

# Local
# NOTE: Subcommands are using local imports to speed up startup time.
from .config import config as config_group
from .data import data as data_group
from .model import model as model_group
from .sysinfo import get_sysinfo
from .taxonomy import taxonomy as taxonomy_group

# 'fork' is unsafe and incompatible with some hardware accelerators.
# Python 3.14 will switch to 'spawn' on all platforms.
multiprocessing.set_start_method(cfg.DEFAULT_MULTIPROCESSING_START_METHOD, force=True)


class ExpandAliasesGroup(click.Group):
    def __init__(self, *args, **kwargs):
        self.aliases = kwargs.pop("aliases", {})
        super().__init__(*args, **kwargs)

    def get_command(self, ctx, cmd_name):
        if cmd_name in self.aliases:
            cmd = self.aliases[cmd_name]["cmd"]
            group = self.aliases[cmd_name]["group"].name
            c = self.aliases[cmd_name]["cmd"].name
            print(
                f"You are using an aliased command, this will be deprecated in a future release. Please consider using `ilab {group} {c}` instead"
            )
            return cmd
        cmd = click.Group.get_command(self, ctx, cmd_name)
        return cmd

    def format_epilog(self, ctx, formatter):
        """Inject our aliases into the help string"""
        if self.aliases:
            formatter.write_paragraph()
            formatter.write_text("Aliases:")
            with formatter.indentation():
                for alias, commands in sorted(self.aliases.items()):
                    formatter.write_text(
                        "{}: {} {}\n".format(
                            alias, commands["group"].name, commands["cmd"].name
                        )
                    )

        super().format_epilog(ctx, formatter)


aliases = {
    "serve": {"group": model_group.model, "cmd": model_group.serve},
    "train": {"group": model_group.model, "cmd": model_group.train},
    "convert": {"group": model_group.model, "cmd": model_group.convert},
    "chat": {"group": model_group.model, "cmd": model_group.chat},
    "test": {"group": model_group.model, "cmd": model_group.test},
    "evaluate": {"group": model_group.model, "cmd": model_group.evaluate},
    "init": {"group": config_group.config, "cmd": config_group.init},
    "download": {"group": model_group.model, "cmd": model_group.download},
    "diff": {"group": taxonomy_group.taxonomy, "cmd": taxonomy_group.diff},
    "generate": {"group": data_group.data, "cmd": data_group.generate},
}


@click.group(cls=ExpandAliasesGroup, aliases=aliases)
@click.option(
    "--config",
    "config_file",
    type=click.Path(),
    default=cfg.DEFAULT_CONFIG,
    show_default=True,
    help="Path to a configuration file.",
)
@click.version_option(package_name="instructlab")
@click.pass_context
# pylint: disable=redefined-outer-name
def ilab(ctx, config_file):
    """CLI for interacting with InstructLab.

    If this is your first time running ilab, it's best to start with `ilab config init` to create the environment.
    """
    cfg.init(ctx, config_file)


ilab.add_command(model_group.model)
ilab.add_command(taxonomy_group.taxonomy)
ilab.add_command(data_group.data)
ilab.add_command(config_group.config)


@ilab.command
def sysinfo():
    """Print system information"""
    for key, value in get_sysinfo().items():
        print(f"{key}: {value}")
