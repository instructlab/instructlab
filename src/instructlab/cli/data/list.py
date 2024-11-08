# SPDX-License-Identifier: Apache-2.0

# Standard
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple
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


def extract_model_and_timestamp(
    filename: str,
) -> Tuple[Optional[str], bool, Optional[str]]:
    """
    Extracts the model name and complete timestamp from a filename.
    """
    # Extract model name and full timestamp
    pattern_model = rf"^(test|messages|train)_(?P<model_name>[\w\-\.]+)_(?P<timestamp>{TIMESTAMP_REGEX})"
    match_model = re.search(pattern_model, filename)

    if match_model:
        model_name = match_model.group("model_name")
        timestamp = match_model.group("timestamp").replace("_", ":").replace("T", " ")
        return (f"{model_name} {timestamp}", False, timestamp)

    # Extract the timestamp
    pattern_time = rf"(skills_train_msgs|node_datasets|knowledge_train_msgs)_(?P<timestamp>{TIMESTAMP_REGEX})"
    match_time = re.search(pattern_time, filename)

    if match_time:
        timestamp = match_time.group("timestamp").replace("_", ":").replace("T", " ")
        return (f"General {timestamp}", True, timestamp)

    return (None, False, None)


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

    headers = ["Dataset", "Created At", "File size"]
    # Check if data is empty and print an empty table if so
    if not data:
        print_table(headers, data)
        return

    grouped_data = defaultdict(list)
    timestamp_to_model = {}
    general_files = []

    # Process model name files e.g. start with test_, train_, messages_
    for item in data:
        filename = item[0]
        group_key, is_general, timestamp = extract_model_and_timestamp(filename)

        if group_key:
            if is_general:
                general_files.append((item, timestamp))
            else:
                grouped_data[group_key].append(item)
                timestamp_to_model[timestamp] = group_key

    # Process General files e.g. start with skills_train_msgs_, node_datasets_
    for item, timestamp in general_files:
        # Attempt to find a model group for the general file's timestamp
        # If no corresponding model file with the same timestamp exists in `timestamp_to_model`,
        # use the default group "General {timestamp}"
        model_group = timestamp_to_model.get(timestamp, f"General {timestamp}")
        grouped_data[model_group].append(item)

    for idx, (group_key, items) in enumerate(
        sorted(grouped_data.items(), key=lambda x: x[0].split(" ", 1)[1], reverse=True)
    ):
        click.echo(f"{os.linesep if idx > 0 else ''}{group_key}")
        items.sort(key=lambda x: x[0])
        print_table(headers, items)
