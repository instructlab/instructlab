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
from instructlab.configuration import DEFAULTS, read_config


class TestLabInit:
    # When using `from X import Y` you need to understand that Y becomes part
    # of your module, so you should use `my_module.Y`` to patch.
    # When using `import X`, you should use `X.Y` to patch.
    # https://docs.python.org/3/library/unittest.mock.html#where-to-patch?
    @patch("instructlab.config.init.Repo.clone_from")
    def test_init_noninteractive(self, mock_clone_from, cli_runner: CliRunner):
        result = cli_runner.invoke(lab.ilab, ["init", "--non-interactive"])
        assert result.exit_code == 0
        assert os.path.exists(DEFAULTS.CONFIG_FILE)
        assert "config.yaml" in os.listdir(DEFAULTS._config_dir)
        mock_clone_from.assert_called_once()

    def test_init_interactive(self, cli_runner: CliRunner):
        result = cli_runner.invoke(lab.ilab, ["init"], input="\nn")
        assert result.exit_code == 0
        assert "config.yaml" in os.listdir(DEFAULTS._config_dir)

    @patch(
        "instructlab.config.init.Repo.clone_from",
        MagicMock(side_effect=GitError("Authentication failed")),
    )
    def test_init_interactive_git_error(self, cli_runner: CliRunner):
        result = cli_runner.invoke(lab.ilab, ["init"], input="\ny")
        assert result.exit_code == 1, "command finished with an unexpected exit code"
        assert "Failed to clone taxonomy repo: Authentication failed" in result.output
        assert "manually run" in result.output

    @patch("instructlab.config.init.Repo.clone_from")
    def test_init_interactive_clone(self, mock_clone_from, cli_runner: CliRunner):
        result = cli_runner.invoke(lab.ilab, ["init"], input="\ny")
        assert result.exit_code == 0
        assert "config.yaml" in os.listdir(DEFAULTS._config_dir)
        mock_clone_from.assert_called_once()

    def test_init_interactive_with_preexisting_nonempty_taxonomy(
        self, cli_runner: CliRunner
    ):
        os.makedirs(f"{DEFAULTS.TAXONOMY_DIR}/contents")
        result = cli_runner.invoke(lab.ilab, ["init"], input="\n")
        assert result.exit_code == 0
        assert "config.yaml" in os.listdir(DEFAULTS._config_dir)
        assert "taxonomy" in os.listdir(DEFAULTS._data_dir)

    def test_init_interactive_with_preexisting_config(self, cli_runner: CliRunner):
        result = cli_runner.invoke(lab.ilab, ["init"], input="non-default-taxonomy\nn")
        assert result.exit_code == 0
        assert "config.yaml" in os.listdir(DEFAULTS._config_dir)
        config = read_config(DEFAULTS.CONFIG_FILE)
        assert config.generate.taxonomy_path == "non-default-taxonomy"

        # second invocation should ask if we want to overwrite - yes, and change taxonomy path
        result = cli_runner.invoke(lab.ilab, ["init"], input="y\ndifferent-taxonomy\nn")
        assert result.exit_code == 0
        assert "config.yaml" in os.listdir(DEFAULTS._config_dir)
        config = read_config(DEFAULTS.CONFIG_FILE)
        assert config.generate.taxonomy_path == "different-taxonomy"

        # third invocation should again ask, but this time don't overwrite
        result = cli_runner.invoke(lab.ilab, ["init"], input="n")
        assert result.exit_code == 0
        assert "config.yaml" in os.listdir(DEFAULTS._config_dir)
        config = read_config(DEFAULTS.CONFIG_FILE)
        assert config.generate.taxonomy_path == "different-taxonomy"
