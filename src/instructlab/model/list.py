# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
from typing import Generator, Tuple
import logging
import os
import time

# Third Party
import click

# First Party
from instructlab import clickext
from instructlab.configuration import DEFAULTS
from instructlab.model.backends.backends import is_model_gguf, is_model_safetensors

logger = logging.getLogger(__name__)

ModelInfo = Tuple[str, str, str]
TableRow = ModelInfo


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
    is_flag=True,
)
def model_list(model_dirs: list[str], list_checkpoints: bool):
    """lists models"""
    data = []
    for directory in model_dirs:
        d = Path(directory)
        if not d.is_dir():
            continue
        for entry in d.iterdir():
            if is_model_gguf(entry):
                data.append(_analyze_gguf(entry))
            elif entry.is_dir():
                data += list(_analyze_dir(entry, list_checkpoints, directory))
    print_table(("Model Name", "Last Modified", "Size"), data)


def convert_bytes_to_human_readable_size(n_bytes: int) -> str:
    """
    Convert bytes size to a human readable string with the appropriate magnitude. (MiB, GiB, ...)
    """
    magnitude = "B"
    f_bytes = float(n_bytes)
    for mag in ("KiB", "MiB", "GiB", "TiB"):
        if f_bytes < 1024:
            break
        magnitude = mag
        f_bytes /= 1024
    return f"{f_bytes:.2f} {magnitude}"


def print_table(headers: TableRow, data: list[TableRow]):
    """
    Displays rows of data in a nicely formatted table.
    """
    column_widths = [
        max(len(str(row[i])) for row in data + [headers]) for i in range(len(headers))
    ]
    # Print separator line between headers and data
    horizontal_lines = ["-" * (width + 2) for width in column_widths]
    joining_line = "+" + "+".join(horizontal_lines) + "+"
    print(joining_line)
    outputs = []
    for header, width in zip(headers, column_widths, strict=False):
        outputs.append(f" {header:{width}} ")
    print("|" + "|".join(outputs) + "|")
    print(joining_line)
    for row in data:
        outputs = []
        for item, width in zip(row, column_widths, strict=False):
            outputs.append(f" {item:{width}} ")
        print("|" + "|".join(outputs) + "|")
    print(joining_line)


def get_model_info(path: str, mtime: float, n_bytes: int) -> ModelInfo:
    return (
        path,
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime)),
        convert_bytes_to_human_readable_size(n_bytes),
    )


def _analyze_gguf(entry: Path) -> ModelInfo:
    stat = entry.stat(follow_symlinks=False)
    return get_model_info(entry.name, stat.st_mtime, stat.st_size)


# There are only two directory cases:
# (1) It's a Safetensors model
# (2) Any other directory (e.g. checkpoint dir `step-19`)
def _analyze_dir(
    entry: Path, list_checkpoints: bool, directory: str
) -> Generator[ModelInfo, None, None]:
    list_all = list_checkpoints and directory == DEFAULTS.CHECKPOINTS_DIR  # (2)
    # TODO: switch to Path.walk after python3.12 bump
    for root_, _, files in os.walk(entry.as_posix()):
        root = Path(root_)
        if not (
            list_all  # (2)
            or is_model_safetensors(root)  # (1)
        ):
            continue

        total_bytes = sum(
            os.stat(os.path.join(root, f), follow_symlinks=False).st_size for f in files
        )

        yield get_model_info(
            # Get the last two parts and join them back into a path
            os.path.join(root.parent.name, root.name),
            entry.stat().st_mtime,
            total_bytes,
        )
