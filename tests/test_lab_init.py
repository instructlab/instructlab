# SPDX-FileCopyrightText: The InstructLab Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from unittest.mock import MagicMock, patch
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @patch("git.Repo.clone_from", MagicMock())
    def test_init(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = CliRunner().invoke(lab.init, ["--non-interactive"])
            self.assertEqual(result.exit_code, 0)

    @patch("git.Repo.clone_from", MagicMock())
    def test_config_pydantic(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            CliRunner().invoke(lab.init, ["--non-interactive"])
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
            result = CliRunner().invoke(lab.init, ["--non-interactive"])
            self.assertEqual(result.exit_code, 1)
