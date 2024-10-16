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
from instructlab.data.list_data import list_data  # type: ignore
from instructlab.utils import print_table

logger = logging.getLogger(__name__)


@click.command(name="list")
@clickext.display_params
@click.option(
    "--dataset-dirs",
    help="Base directories where datasets are stored.",
    multiple=True,
    default=[DEFAULTS.DATASETS_DIR],
    show_default=True,
)
def list_datasets(dataset_dirs):
    """lists datasets"""

    data: List[List[Path]] = []

    dirs = [Path(dir) for dir in dataset_dirs]

    try:
        data = list_data(dirs)
    except OSError as exc:
        click.secho(f"Failed to list datasets with exception: {exc}")
        raise click.exceptions.Exit(1)

    headers = ["Dataset", "Created At", "File size"]
    print_table(headers, data)
