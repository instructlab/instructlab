# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
from typing import List
import logging

# Third Party
import click

# First Party
from instructlab import clickext
from instructlab.configuration import DEFAULTS
from instructlab.utils import list_models, print_table

logger = logging.getLogger(__name__)


@click.command(name="list")
@clickext.display_params
@click.option(
    "--model-dirs",
    help="Base directories where models are stored.",
    multiple=True,
    default=[DEFAULTS.MODELS_DIR],
    show_default=True,
)
@click.option(
    "--list-checkpoints",
    help="Also list checkpoints [in addition to existing models].",
    is_flag=True,
)
def model_list(model_dirs: List[str], list_checkpoints: bool):
    """Lists models"""
    # click converts lists into tuples when using multiple
    model_path_dirs = [Path(m) for m in model_dirs]

    data = list_models(model_path_dirs, list_checkpoints)
    data_as_lists = [list(item) for item in data]  # this is to satisfy mypy
    print_table(["Model Name", "Last Modified", "Size"], data_as_lists)
