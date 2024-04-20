# SPDX-License-Identifier: Apache-2.0

# Standard
from unittest import mock
import sys

# Third Party
import pytest

# Local
from .taxonomy import MockTaxonomy


@pytest.fixture
def taxonomy_dir(tmp_path):
    with MockTaxonomy(tmp_path) as taxonomy:
        yield taxonomy


@pytest.fixture(scope="class")
def mock_mlx_package():
    """Add mocked 'mlx' modules to sys.path

    The 'mlx' package is only available on Apple Silicon. This fixture
    patches sys.modules so local imports and mock-patching of 'mlx' attributes
    works.
    """
    mlx_modules = {
        name: mock.MagicMock()
        for name in ["mlx", "mlx.core", "mlx.nn", "mlx.optimizers", "mlx.utils"]
    }
    with mock.patch.dict(sys.modules, mlx_modules):
        yield
