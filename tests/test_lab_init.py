# Standard
from unittest.mock import MagicMock, patch
import os
import unittest

# Third Party
from click.testing import CliRunner
from git import GitError
import pydantic
import pydantic_yaml

# First Party
from cli import lab
from tests.schema import Config


class TestLabInit(unittest.TestCase):
    @patch("git.Repo.clone_from", MagicMock())
    def test_init(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(lab.init, ["--non-interactive"])
            self.assertEqual(result.exit_code, 0)
            self.assertTrue("config.yaml" in os.listdir())

    @patch("git.Repo.clone_from", MagicMock())
    def test_config_pydantic(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            runner.invoke(lab.init, ["--non-interactive"])
            try:
                pydantic_yaml.parse_yaml_file_as(model_type=Config, file="config.yaml")
                self.assertTrue
            except TypeError as e:
                print(e)
                self.assertFalse
            except pydantic.ValidationError as e:
                print(e)
                assert self.assertFalse

    @patch(
        "git.Repo.clone_from", MagicMock(side_effect=GitError("Authentication failed"))
    )
    def test_init_git_error(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(lab.init, ["--non-interactive"])
            self.assertEqual(result.exit_code, 1)
            self.assertTrue(
                "Failed to clone taxonomy repo: Authentication failed" in result.output
            )
