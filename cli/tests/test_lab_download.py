from click.testing import CliRunner
import pytest
import cli.lab as lab
from contextlib import ExitStack
from functools import partial

def test_download():
    runner = CliRunner()
    lab_test = lab
    with runner.isolated_filesystem():
        result = runner.invoke(lab_test.download, ["--repository", "https://github.com/instruct-lab/cli.git", "--model-dir", "models", "--pattern", "*"])
        assert result.exit_code == 0   

    result = runner.invoke(lab.download)
    assert result.exit_code == 0 

def test_download_invalid():
    with ExitStack() as stack:
        runner = CliRunner()
        lab_test = lab
        exec("mkdir", "temp")
        stack.callback(partial(exec("rmdir", "temp")))
        with runner.isolated_filesystem():
            # should fail due to models dir already existing
            result = runner.invoke(lab_test.download, ["--model-dir", "temp"])
            assert result.exit_code == 1
            # should fail due to invalid pattern
            result = runner.invoke(lab_test.download, ["--pattern", "foobar"])
            assert result.exit_code == 1
            # should fail due to invalid URL
            result = runner.invoke(lab_test.download, ["--repository", "foobar"])
            assert result.exit_code == 1
