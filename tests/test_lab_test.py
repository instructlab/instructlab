import logging
import pytest
from click.testing import CliRunner
from cli import lab
from pathlib import Path
from shutil import copytree, rmtree

## Files required by lab test:
# <model_dir>/adapters.npz
# <data_dir>/test.jsonl (utf-8)
TESTING_DATA_PATH = Path(__file__).parent.resolve() / "testing_data" / "lab_test"

@pytest.fixture()
def setup_testing_env():
    runner = CliRunner()
    with runner.isolated_filesystem() as fs:
        testing_env = Path(fs)
        copytree(TESTING_DATA_PATH, testing_env, dirs_exist_ok=True)
        yield runner, testing_env


def test_lab_test(setup_testing_env):
    runner, _ = setup_testing_env
    result = runner.invoke(lab.test)
    logging.debug("TEST DONE")
    assert result.exit_code == 0
