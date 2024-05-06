# SPDX-License-Identifier: Apache-2.0

# Standard
from unittest.mock import MagicMock, patch
import os
import unittest

# Third Party
from click.testing import CliRunner
from git import GitError

# First Party
from instructlab import lab
from instructlab.config import read_config


class TestLabInit(unittest.TestCase):
    # When using `from X import Y` you need to understand that Y becomes part
    # of your module, so you should use `my_module.Y`` to patch.
    # When using `import X`, you should use `X.Y` to patch.
    # https://docs.python.org/3/library/unittest.mock.html#where-to-patch?
    @patch("instructlab.lab.Repo.clone_from")
    def test_init_noninteractive(self, mock_clone_from):
        result = CliRunner().invoke(lab.init, args=["--non-interactive"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("config.yaml", os.listdir())
        mock_clone_from.assert_called_once()

    def test_init_interactive(self):
        result = CliRunner().invoke(lab.init, input="\nn")
        self.assertEqual(result.exit_code, 0)
        self.assertIn("config.yaml", os.listdir())

    @patch(
        "instructlab.lab.Repo.clone_from",
        MagicMock(side_effect=GitError("Authentication failed")),
    )
    def test_init_interactive_git_error(self):
        result = CliRunner().invoke(lab.init, input="\ny")
        self.assertEqual(
            result.exit_code, 1, "command finished with an unexpected exit code"
        )
        self.assertIn(
            "Failed to clone taxonomy repo: Authentication failed", result.output
        )
        self.assertIn("manually run", result.output)

    @patch("instructlab.lab.Repo.clone_from")
    def test_init_interactive_clone(self, mock_clone_from):
        result = CliRunner().invoke(lab.init, input="\ny")
        self.assertEqual(result.exit_code, 0)
        self.assertIn("config.yaml", os.listdir())
        mock_clone_from.assert_called_once()

    def test_init_interactive_with_preexisting_nonempty_taxonomy(self):
        os.makedirs("taxonomy/contents")
        result = CliRunner().invoke(lab.init, input="\n")
        self.assertEqual(result.exit_code, 0)
        self.assertIn("config.yaml", os.listdir())
        self.assertIn("taxonomy", os.listdir())

    def test_init_interactive_with_preexisting_config(self):
        runner = CliRunner()
        # first run to prime the config.yaml in current directory
        result = runner.invoke(lab.init, input="non-default-taxonomy\nn")
        self.assertEqual(result.exit_code, 0)
        self.assertIn("config.yaml", os.listdir())
        config = read_config("config.yaml")
        self.assertEqual(config.generate.taxonomy_path, "non-default-taxonomy")

        # second invocation should ask if we want to overwrite - yes, and change taxonomy path
        result = runner.invoke(lab.init, input="y\ndifferent-taxonomy\nn")
        self.assertEqual(result.exit_code, 0)
        self.assertIn("config.yaml", os.listdir())
        config = read_config("config.yaml")
        self.assertEqual(config.generate.taxonomy_path, "different-taxonomy")

        # third invocation should again ask, but this time don't overwrite
        result = runner.invoke(lab.init, input="n")
        self.assertEqual(result.exit_code, 0)
        self.assertIn("config.yaml", os.listdir())
        config = read_config("config.yaml")
        self.assertEqual(config.generate.taxonomy_path, "different-taxonomy")
