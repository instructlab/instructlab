# SPDX-License-Identifier: Apache-2.0

# Standard
from collections import defaultdict
from pathlib import Path
from typing import List
import logging
import os
import re

# Third Party
import click

# First Party
from instructlab import clickext
from instructlab.configuration import DEFAULTS
from instructlab.data.list_data import list_data  # type: ignore
from instructlab.utils import print_table

logger = logging.getLogger(__name__)

# A constant for the timestamp regex pattern
TIMESTAMP_REGEX = r"\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}"


def extract_model_name(
    filename: str,
) -> str:
    """
    Extracts the model name from a filename, or "General" if not applicable.
    """
    # Extract model name
    pattern_model = rf"^(test|messages|train)_(?P<model_name>[\w\-\.]+)_(?P<timestamp>{TIMESTAMP_REGEX})"
    match_model = re.search(pattern_model, filename)

    if match_model:
        model_name = match_model.group("model_name")
        return f"{model_name}"

    return "General"


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

    data: List[List[str]] = []

    dirs = [Path(dir) for dir in dataset_dirs]

    try:
        data = list_data(dirs)
    except OSError as exc:
        click.secho(f"Failed to list datasets with exception: {exc}")
        raise click.exceptions.Exit(1)

    headers = ["Dataset", "Model", "Created At", "File size"]
    # Check if data is empty and print an empty table if so
    if not data:
        print_table(headers, data)
        return

    grouped_data = defaultdict(list)

    # Process model name files e.g. start with test_, train_, messages_
    for item in data:
        filename = item[0]
        created_at = item[1]
        size = item[2]
        run_id = item[3]
        model_name = extract_model_name(os.path.basename(filename))

        grouped_data[run_id].append(
            [
                filename,
                model_name,
                created_at,
                size,
            ]
        )

    for idx, run_id in enumerate(sorted(grouped_data.keys(), reverse=True)):
        click.echo(
            f"{os.linesep if idx > 0 else ''}{'Uncategorized datasets' if run_id == '' else f'Run from {run_id}'}"
        )
        item = grouped_data[run_id]
        item.sort(key=lambda x: x[0])
        print_table(headers, item)
