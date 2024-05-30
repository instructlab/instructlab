# SPDX-License-Identifier: Apache-2.0

# pylint: disable=too-many-lines

# Standard
from glob import glob
from os.path import basename, dirname, exists
from pathlib import Path
import json
import logging
import multiprocessing
import os
import shutil
import sys
import typing

# Third Party
from click_didyoumean import DYMGroup
from git import GitError, Repo
from huggingface_hub import hf_hub_download, list_repo_files
from huggingface_hub import logging as hf_logging
from huggingface_hub import snapshot_download
import click
import yaml

# Local
# NOTE: Subcommands are using local imports to speed up startup time.
from . import config, log, utils
from .sysinfo import get_sysinfo

# 'fork' is unsafe and incompatible with some hardware accelerators.
# Python 3.14 will switch to 'spawn' on all platforms.
multiprocessing.set_start_method(
    config.DEFAULT_MULTIPROCESSING_START_METHOD, force=True
)

# Set logging level of OpenAI client and httpx library to ERROR to suppress INFO messages
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

if typing.TYPE_CHECKING:
    # Third Party
    import torch


@click.group(cls=DYMGroup)
@click.option(
    "--config",
    "config_file",
    type=click.Path(),
    default=config.DEFAULT_CONFIG,
    show_default=True,
    help="Path to a configuration file.",
)
@click.version_option(package_name="instructlab")
@click.pass_context
# pylint: disable=redefined-outer-name
def ilab(ctx, config_file):
    """CLI for interacting with InstructLab.

    If this is your first time running InstructLab, it's best to start with `ilab init` to create the environment.
    """
    # ilab init or "--help" have no config file. ilab sysinfo does not need one.
    # CliRunner does fill ctx.invoke_subcommand in option callbacks. We have
    # to validate config_file here.
    config.init_config(ctx, config_file)


@ilab.command()
@click.option(
    "--taxonomy-path",
    type=click.Path(),
    help=f"Path to {config.DEFAULT_TAXONOMY_REPO} clone or local file path.",
)
@click.option(
    "--taxonomy-base",
    help="Base git-ref to use for taxonomy.",
)
@click.option(
    "--yaml-rules",
    type=click.Path(),
    default=None,
    help="Custom rules file for YAML linting.",
)
@click.option(
    "--quiet",
    is_flag=True,
    help="Suppress all output. Call returns 0 if check passes, 1 otherwise.",
)
@click.pass_context
def diff(ctx, taxonomy_path, taxonomy_base, yaml_rules, quiet):
    """
    Lists taxonomy files that have changed since <taxonomy-base>
    and checks that taxonomy is valid. Similar to 'git diff <ref>'.
    """
    # pylint: disable=C0415
    # Local
    from .utils import get_taxonomy_diff, read_taxonomy

    if not taxonomy_base:
        taxonomy_base = ctx.obj.config.generate.taxonomy_base
    if not taxonomy_path:
        taxonomy_path = ctx.obj.config.generate.taxonomy_path
    if not ctx.obj:
        logger = logging.getLogger(__name__)
    else:
        logger = ctx.obj.logger

    if not quiet:
        is_file = os.path.isfile(taxonomy_path)
        if is_file:  # taxonomy_path is file
            click.echo(taxonomy_path)
        else:  # taxonomy_path is dir
            try:
                updated_taxonomy_files = get_taxonomy_diff(taxonomy_path, taxonomy_base)
            except (SystemExit, GitError) as exc:
                click.secho(
                    f"Reading taxonomy failed with the following error: {exc}",
                    fg="red",
                )
                raise SystemExit(1) from exc
            for f in updated_taxonomy_files:
                click.echo(f)
    try:
        read_taxonomy(logger, taxonomy_path, taxonomy_base, yaml_rules)
    except (SystemExit, yaml.YAMLError) as exc:
        if not quiet:
            click.secho(
                f"Reading taxonomy failed with the following error: {exc}",
                fg="red",
            )
        raise SystemExit(1) from exc
    if not quiet:
        click.secho(
            f"Taxonomy in /{taxonomy_path}/ is valid :)",
            fg="green",
        )


# ilab list => ilab diff
# ilab check => ilab diff --quiet
utils.make_lab_diff_aliases(ilab, diff)


@ilab.command
def sysinfo():
    """Print system information"""
    for key, value in get_sysinfo().items():
        print(f"{key}: {value}")
