from click.testing import CliRunner
import pytest
import cli.lab as lab

def test_download():
    runner = CliRunner()
    lab_test = lab
    with runner.isolated_filesystem():
        result = runner.invoke(lab_test.download, ["--repository", "https://github.com/instruct-lab/cli.git", "--model-dir", "models", "--pattern", "*"])
        assert result.exit_code == 0   

    result = runner.invoke(lab.download)
    assert result.exit_code == 0 

def test_download_invalid():
    runner = CliRunner()
    lab_test = lab
    exec("mkdir", "tmp")

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
