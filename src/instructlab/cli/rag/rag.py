# SPDX-License-Identifier: Apache-2.0

# This code instantiates the CLI ilab rag group.

# Standard

# Third Party
import click

# First Party
from instructlab import clickext
from instructlab.configuration import storage_dirs_exist


@click.group(cls=clickext.LazyEntryPointGroup, ep_group="instructlab.command.rag")
@click.pass_context
@clickext.display_params
def rag(ctx):
    """Command Group for Interacting with the RAG for InstructLab.
    If this is your first time running ilab, it's best to start with `ilab config init` to create the environment.
    """
    ctx.obj = ctx.parent.obj
    ctx.obj.ensure_config(ctx)
    if not storage_dirs_exist():
        click.secho(
            "Some ilab storage directories do not exist yet. Please run `ilab config init` before continuing.",
            fg="red",
        )
        raise click.exceptions.Exit(1)
    ctx.default_map = ctx.parent.default_map
