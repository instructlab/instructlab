# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
from typing import List

# First Party
from instructlab.utils import list_models, print_table


def list_and_print_models(model_path_dirs: List[Path], list_checkpoints: bool):
    """
    Core functionality for listing models and optionally listing checkpoints.
    """
    data = list_models(model_path_dirs, list_checkpoints)
    data_as_lists = [list(item) for item in data]  # this is to satisfy mypy
    print_table(["Model Name", "Last Modified", "Size"], data_as_lists)
