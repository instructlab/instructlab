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
import click

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


@ilab.command
def sysinfo():
    """Print system information"""
    for key, value in get_sysinfo().items():
        print(f"{key}: {value}")
