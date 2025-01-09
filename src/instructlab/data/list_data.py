# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
from typing import List
import logging
import time

# First Party
from instructlab.utils import convert_bytes_to_proper_mag

logger = logging.getLogger(__name__)


def list_data(dataset_dirs: list[str]) -> List[List[str]]:
    """
    Recursively lists all JSONL files in the given directories and gathers their metadata.

    Args:
        dataset_dirs (list[Path]): A list of paths to directories where JSONL files are searched for.

    Returns:
        List[List[Path]]: A list of lists, where each inner list contains three strings:
            - The relative file path of the `.jsonl` file.
            - The creation timestamp of the file (format: 'YYYY-MM-DD HH:MM:SS').
            - The formatted size of the file with appropriate units (e.g., '12.34 MB').

    Raises:
        OSError: If any directory in `dataset_dirs` does not exist.
    """

    data: List[List[str]] = []
    for directory in dataset_dirs:
        dirpath = Path(directory)
        directories: List[Path] = [dirpath]
        top_level_dir = dirpath
        if not dirpath.exists():
            raise OSError(f"directory does not exist: {dirpath}")
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

                # in some cases, data will be categorized per run
                # parse top-level subdirectory to determine run name
                run_id = Path(relative_name).parts[0]

                data.append(
                    [
                        relative_name,
                        created_at,
                        formatted_size,
                        run_id,
                    ]
                )

    return data
