# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest

# Local
from .taxonomy import MockTaxonomy


@pytest.fixture
def taxonomy_dir(tmp_path):
    with MockTaxonomy(tmp_path) as taxonomy:
        yield taxonomy
