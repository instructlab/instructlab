# SPDX-License-Identifier: Apache-2.0

# Standard
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
from instructlab.utils import (
    convert_bytes_to_proper_mag,
    get_model_metadata,
    list_models,
    print_table,
)

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
    print(f"All listed models can be found at: {DEFAULTS.MODELS_DIR}")

    # click converts lists into tuples when using multiple
    model_path_dirs = [Path(m) for m in model_dirs]

    valid_models = list_models(model_path_dirs, list_checkpoints)

    # populate model_info as a dictionary with model name as key
    # and value as a list of dictionaries. Each dictionary represents
    # metadata info for a specific version of the model.
    model_info: dict[str, List[dict]] = {}
    for model in valid_models:
        model_metadata, file_found = get_model_metadata(Path(model[0]))

        # calculate model name as the model's path relative to ~/.cache/instructlab/models
        model_name = os.path.relpath(model[0], model[1])

        # remove version subfolder from model name if metadata
        # file was found
        if file_found:
            model_name = re.sub(r"/[^/]*$", "", model_name)

        if model_name not in model_info:
            model_info[model_name] = []

        adjusted_all_sizes, magnitude = convert_bytes_to_proper_mag(
            int(model_metadata["size"])
        )
        model_metadata["size"] = f"{adjusted_all_sizes:.1f} {magnitude}"
        model_info[model_name].append(model_metadata)

    # convert model_info into a list of lists to feed it into the table
    # each unique model version becomes its own list
    model_info_as_lists: List[List[str]] = []
    for key, value in model_info.items():
        # populate the list for the first version of any model with the model name
        # this will represent the first row for that model in the table
        metadata_list = [list(dict.values()) for dict in value]
        metadata_list[0].insert(0, key)

        # subsequent versions of that model will have empty string as model name to leave a gap
        # in the table
        for sublist in metadata_list[1:]:
            sublist.insert(0, "")

        model_info_as_lists += metadata_list

    print_table(
        ["Model Name", "Size", "Last Modified", "Version", "SHA"], model_info_as_lists
    )
