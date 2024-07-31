# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
import logging
import os
import time

# Third Party
import click

# First Party
from instructlab import clickext
from instructlab.configuration import DEFAULTS
from instructlab.model.backends.backends import is_model_gguf, is_model_safetensors
from instructlab.utils import convert_bytes_to_proper_mag, print_table

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
    is_flag=True,
)
def model_list(model_dirs: list[str], list_checkpoints: bool):
    """Lists models"""
    data = []
    for directory in model_dirs:
        for entry in Path(directory).iterdir():
            # if file, just tally the size. This must be a GGUF.
            if entry.is_file() and is_model_gguf(entry):
                model, age, size = _analyze_gguf(entry)
                data.append([model, age, size])
            elif entry.is_dir():
                for m in _analyze_dir(entry, list_checkpoints, directory):
                    data.append(m)
    print_table(["Model Name", "Last Modified", "Size"], data)


# convert_bytes_to_proper_mag takes a dir/file size in Bytes and converts it to the proper Magnitude (MiB, GiB)


AnalyzeResultGGUF = list[str]
AnalyzeResultDir = list[list[str]]


def _analyze_gguf(entry: Path) -> AnalyzeResultGGUF:
    # stat the gguf, add it to the table
    stat = Path(entry.absolute()).stat(follow_symlinks=False)
    f_size = stat.st_size
    adjusted_size, magnitude = convert_bytes_to_proper_mag(f_size)
    # add to table
    modification_time = os.path.getmtime(entry.absolute())
    modification_time_readable = time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(modification_time)
    )
    return [entry.name, modification_time_readable, f"{adjusted_size:.1f} {magnitude}"]


def _analyze_dir(
    entry: Path, list_checkpoints: bool, directory: str
) -> AnalyzeResultDir:
    actual_model_name = ""
    all_files_sizes = 0
    add_model = False
    models = []
    # walk entire dir.
    for root, _, files in os.walk(entry.as_posix()):
        normalized_path = os.path.normpath(root)
        # Split the path into its components
        parts = normalized_path.split(os.sep)
        # Get the last two parts and join them back into a path
        last_two_parts = os.path.join(parts[-2], parts[-1])
        # if this is a dir it could be:
        # top level repo dir `instructlab/`
        # top level model dir `instructlab/granite-7b-lab`
        # checkpoint top level dir `step-19`
        # any lower level dir: `instructlab/granite-7b-lab/.huggingface/download.....`
        # so, check if model is valid Safetensor, GGUF, or list it regardless w/ `--list-checkpoints`
        # if --list-checkpoints is specified, we will list all checkpoints in the checkpoints dir regardless of the validity
        if is_model_safetensors(Path(normalized_path)) or is_model_gguf(
            Path(normalized_path)
        ):
            actual_model_name = last_two_parts
            all_files_sizes = 0
            add_model = True
        else:
            if list_checkpoints and directory is DEFAULTS.CHECKPOINTS_DIR:
                logging.debug("Including model regardless of model validity")
            else:
                continue
        for f in files:
            # stat each file in the dir, add the size in Bytes, then convert to proper magnitude
            full_file = os.path.join(root, f)
            stat = Path(full_file).stat()
            all_files_sizes += stat.st_size
        adjusted_all_sizes, magnitude = convert_bytes_to_proper_mag(all_files_sizes)
        if add_model:
            # add to table
            modification_time = os.path.getmtime(entry.absolute())
            modification_time_readable = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(modification_time)
            )
            models.append(
                [
                    actual_model_name,
                    modification_time_readable,
                    f"{adjusted_all_sizes:.1f} {magnitude}",
                ]
            )
    return models
