# SPDX-License-Identifier: Apache-2.0

# Standard
from unittest import mock
import os
import re
import shutil
import sys

# Third Party
from click.testing import CliRunner
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


_SKIP_FILESYSTEM_ISOLATION_TESTS = [
    # These cases expect notebooks present in current dir.
    # We could probably copy them into isolated environment too...
    r"^test_notebooks\[.*\]$",
]

_SKIP_TESTS_RE = [re.compile(pattern) for pattern in _SKIP_FILESYSTEM_ISOLATION_TESTS]


def should_skip_filesystem_isolation(testname):
    return any(re.match(p, testname) for p in _SKIP_TESTS_RE)


def get_testdata_dir():
    return os.path.join(os.getcwd(), "tests/testdata")


@pytest.fixture(scope="function", autouse=True)
def isolate_filesystem(request):
    if should_skip_filesystem_isolation(request.node.name):
        yield
        return

    test_data = get_testdata_dir()
    with CliRunner().isolated_filesystem():
        shutil.copytree(test_data, get_testdata_dir())
        yield
