# SPDX-License-Identifier: Apache-2.0
# pylint: disable=redefined-outer-name

# Standard
from unittest import mock
import os
import pathlib
import sys
import typing

# Third Party
from click.testing import CliRunner
import pytest

# First Party
from instructlab.configuration import DEFAULTS

# Local
from .taxonomy import MockTaxonomy

TESTS_PATH = pathlib.Path(__file__).parent.absolute()


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


@pytest.fixture
def tmp_path_home(tmp_path: pathlib.Path) -> typing.Generator[pathlib.Path, None, None]:
    """Reset $HOME to tmp_path and unset $XDG_*

    Yields `tmp_path` fixture path.
    """
    with mock.patch.dict(os.environ):
        os.environ["HOME"] = str(tmp_path)
        for key in list(os.environ):
            if key.startswith("XDG_"):
                os.environ.pop(key)

        DEFAULTS._reset()  # resets the config defaults to use the new $HOME
        yield tmp_path


@pytest.fixture
def cli_runner(
    tmp_path_home: pathlib.Path,
) -> typing.Generator[CliRunner, None, None]:
    """Click CLI runner

    Run click's CliRunner with file system isolation (chdir), $HOME reset,
    and $XDG_* unset. Yields runner instance with its temp dir inside
    `tmp_path` fixture path.
    """
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path_home):
        yield runner


@pytest.fixture
def testdata_path() -> typing.Generator[pathlib.Path, None, None]:
    """Path to local test data directory"""
    yield TESTS_PATH / "testdata"
