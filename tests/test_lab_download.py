from click.testing import CliRunner
import pytest
import cli.lab as lab
from contextlib import ExitStack
from functools import partial
import os

def test_download():
    runner = CliRunner()
    lab_test = lab
    with runner.isolated_filesystem():
        result = runner.invoke(lab.init, ["--interactive"])
        assert result.exit_code == 0  
        result = runner.invoke(lab.download, ["--hf-endpoint", "http://localhost:8080/", "--repository", "https://github.com/instruct-lab/cli.git", "--model-dir", "models"])
        assert result.exit_code == 0   

    result = runner.invoke(lab.download)
    assert result.exit_code == 0 

def test_download_invalid():
    with ExitStack() as stack:
        runner = CliRunner()
        lab_test = lab
        os.mkdir("temp")
        stack.callback(partial(os.rmdir, "temp"))
        with runner.isolated_filesystem():
            result = runner.invoke(lab.init, ["--interactive"])
            assert result.exit_code == 0  
            # should fail due to invalid pattern
            result = runner.invoke(lab_test.download, ["--hf-endpoint", "http://localhost:8080/", "--release", "foobar"])
            assert result.exit_code == 1
            # should fail due to invalid URL
            result = runner.invoke(lab_test.download, ["--hf-endpoint", "http://localhost:8080/", "--repository", "foobar"])
            assert result.exit_code == 1
