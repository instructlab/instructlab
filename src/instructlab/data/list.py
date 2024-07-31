# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
from typing import List, TypedDict
import logging
import time

# Third Party
import click

# First Party
from instructlab import clickext
from instructlab.configuration import DEFAULTS
from instructlab.utils import convert_bytes_to_proper_mag, print_table

logger = logging.getLogger(__name__)


class DatasetListing(TypedDict):
    """
    Represents a single dataset listing.
    """

    filename: str
    created_at: str
    file_size: str


@click.command(name="list")
@clickext.display_params
@click.option(
    "--dataset-dirs",
    help="Base directories where datasets are stored.",
    multiple=True,
    default=[DEFAULTS.DATASETS_DIR],
    show_default=True,
)
@click.pass_context
def list_data(ctx, dataset_dirs: list[str]):
    """lists datasets"""
    data: List[List[str]] = []
    for directory in dataset_dirs:
        dirpath = Path(directory)
        directories: List[Path] = [dirpath]
        top_level_dir = dirpath
        if not dirpath.exists():
            ctx.fail(f"directory does not exist: {dirpath}")
        while len(directories) > 0:
            current_dir = directories.pop()
            for entry in current_dir.iterdir():
                if entry.is_dir():
                    directories.append(entry)
                    continue
                if entry.suffix != ".jsonl":
                    continue

                stat = entry.stat()
                created_at = time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(stat.st_ctime)
                )
                fsize, mag = convert_bytes_to_proper_mag(stat.st_size)
                formatted_size = f"{fsize:.2f} {mag}"

                # capture the entries as 'messages.jsonl', 'dir/dataset2.jsonl'
                # with respect to the top-level directory
                relative_name = str(entry.relative_to(top_level_dir))
                data.append(
                    [
                        relative_name,
                        created_at,
                        formatted_size,
                    ]
                )

    headers = ["Dataset", "Created At", "File size"]
    print_table(headers, data)
