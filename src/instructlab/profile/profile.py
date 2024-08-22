# SPDX-License-Identifier: Apache-2.0

# Third Party
from click_didyoumean import DYMGroup
import click


# First Party
from instructlab import clickext
from instructlab.configuration import storage_dirs_exist

@click.group(
    cls=clickext.LazyEntryPointGroup,
    ep_group="instructlab.command.profile",
)
@click.pass_context
# pylint: disable=redefined-outer-name
def profile(ctx):
    """Command Group for Interacting with performance Profiles in instructlab.

    If this is your first time running ilab, it's best to start with `ilab config init` to create the environment.
    """
    ctx.parent.obj.ensure_config(ctx)
    if not storage_dirs_exist():
        click.secho(
            "Some ilab storage directories do not exist yet. Please run `ilab config init` before continuing.",
            fg="red",
        )
        raise click.exceptions.Exit(1)
    ctx.obj = ctx.parent.obj
    ctx.default_map = ctx.parent.default_map