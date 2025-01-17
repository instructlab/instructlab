# SPDX-License-Identifier: Apache-2.0

# Some basic utility methods for pulling documents out of an InstructLab taxonomy.

# Standard
from pathlib import Path
import logging

# Third Party
from instructlab.sdg.utils.taxonomy import read_taxonomy_leaf_nodes

logger = logging.getLogger(__name__)


def lookup_knowledge_files(taxonomy_path, taxonomy_base, temp_dir) -> list[Path]:
    """
    Lookup updated or new knowledge files in the taxonomy repo.
    Download the documents referenced in the configured datasets folder under a temporary folder.
    Finally, groups all the documents at the root of this folder and returns the list of paths.
    """
    yaml_rules = None
    leaf_nodes = read_taxonomy_leaf_nodes(
        taxonomy_path, taxonomy_base, yaml_rules, temp_dir
    )
    knowledge_files: list[Path] = []
    for leaf_node in leaf_nodes.values():
        knowledge_files.extend(leaf_node[0]["filepaths"])

    grouped_knowledge_files = []
    for knowledge_file in knowledge_files:
        grouped_knowledge_files.append(
            knowledge_file.rename(Path(temp_dir) / knowledge_file.name)
        )

    return knowledge_files
