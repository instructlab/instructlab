# SPDX-License-Identifier: Apache-2.0
# pylint: disable=duplicate-code

# Standard
from unittest.mock import MagicMock, patch
import os

# Third Party
from click.testing import CliRunner
from git import GitError

# First Party
from instructlab import lab
from instructlab.configuration import read_config


class TestLabInit:
    # When using `from X import Y` you need to understand that Y becomes part
    # of your module, so you should use `my_module.Y`` to patch.
    # When using `import X`, you should use `X.Y` to patch.
    # https://docs.python.org/3/library/unittest.mock.html#where-to-patch?
    @patch("instructlab.config.init.Repo.clone_from")
    def test_init_noninteractive(self, mock_clone_from):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(lab.ilab, ["init", "--non-interactive"])
            assert result.exit_code == 0
            assert "config.yaml" in os.listdir()
            mock_clone_from.assert_called_once()

    def test_init_interactive(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(lab.ilab, ["init"], input="\nn")
            assert result.exit_code == 0
            assert "config.yaml" in os.listdir()

    @patch(
        "instructlab.config.init.Repo.clone_from",
        MagicMock(side_effect=GitError("Authentication failed")),
    )
    def test_init_interactive_git_error(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(lab.ilab, ["init"], input="\ny")
            assert (
                result.exit_code == 1
            ), "command finished with an unexpected exit code"
            assert (
                "Failed to clone taxonomy repo: Authentication failed" in result.output
            )
            assert "manually run" in result.output

    @patch("instructlab.config.init.Repo.clone_from")
    def test_init_interactive_clone(self, mock_clone_from):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(lab.ilab, ["init"], input="\ny")
            assert result.exit_code == 0
            assert "config.yaml" in os.listdir()
            mock_clone_from.assert_called_once()

    def test_init_interactive_with_preexisting_nonempty_taxonomy(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            os.makedirs("taxonomy/contents")
            result = runner.invoke(lab.ilab, ["init"], input="\n")
            assert result.exit_code == 0
            assert "config.yaml" in os.listdir()
            assert "taxonomy" in os.listdir()

    def test_init_interactive_with_preexisting_config(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            # first run to prime the config.yaml in current directory
            result = runner.invoke(lab.ilab, ["init"], input="non-default-taxonomy\nn")
            assert result.exit_code == 0
            assert "config.yaml" in os.listdir()
            config = read_config("config.yaml")
            assert config.generate.taxonomy_path == "non-default-taxonomy"

            # second invocation should ask if we want to overwrite - yes, and change taxonomy path
            result = runner.invoke(lab.ilab, ["init"], input="y\ndifferent-taxonomy\nn")
            assert result.exit_code == 0
            assert "config.yaml" in os.listdir()
            config = read_config("config.yaml")
            assert config.generate.taxonomy_path == "different-taxonomy"

            # third invocation should again ask, but this time don't overwrite
            result = runner.invoke(lab.ilab, ["init"], input="n")
            assert result.exit_code == 0
            assert "config.yaml" in os.listdir()
            config = read_config("config.yaml")
            assert config.generate.taxonomy_path == "different-taxonomy"
