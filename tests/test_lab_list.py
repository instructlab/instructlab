# Standard
from unittest.mock import MagicMock, patch
import unittest

# Third Party
from click.testing import CliRunner
from git import NoSuchPathError
import pytest

# First Party
from cli import lab
from cli.generator.utils import GenerateException

TAXONOMY_BASE = "master"


class TestLabList(unittest.TestCase):
    """Test collection for `lab list` command."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @pytest.fixture(autouse=True)
    def _init_taxonomy(self, taxonomy_dir):
        self.taxonomy = taxonomy_dir

    def test_list(self):
        untracked_file = "compositional_skills/new/qna.yaml"
        tracked_file = "compositional_skills/tracked/qna.yaml"
        self.taxonomy.create_untracked(untracked_file)
        self.taxonomy.add_tracked(tracked_file)
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(
                lab.list,
                [
                    "--taxonomy-base",
                    TAXONOMY_BASE,
                    "--taxonomy-path",
                    self.taxonomy.root,
                ],
            )
            self.assertIn(untracked_file, result.output)
            self.assertNotIn(tracked_file, result.output)
            self.assertEqual(result.exit_code, 0)

    def test_list_rm_tracked(self):
        tracked_file = "compositional_skills/tracked/qna.yaml"
        self.taxonomy.add_tracked(tracked_file)
        self.taxonomy.remove_file(tracked_file)
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(
                lab.list,
                [
                    "--taxonomy-base",
                    TAXONOMY_BASE,
                    "--taxonomy-path",
                    self.taxonomy.root,
                ],
            )
            self.assertNotIn(tracked_file, result.output)
            self.assertEqual(result.exit_code, 0)

    @pytest.mark.xfail(
        reason="Invalid extensions are filtered within `get_taxonomy_diff()`, "
        "before the check happens",
        strict=True,
    )
    def test_list_invalid_ext(self):
        untracked_file = "compositional_skills/writing/new/qna.YAML"
        self.taxonomy.create_untracked(untracked_file)
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(
                lab.list,
                [
                    "--taxonomy-base",
                    TAXONOMY_BASE,
                    "--taxonomy-path",
                    self.taxonomy.root,
                ],
            )
            self.assertListEqual(self.taxonomy.untracked_files, [untracked_file])
            self.assertIn("WARNING", result.output)
            self.assertIn(untracked_file, result.output)
            self.assertEqual(result.exit_code, 0)

    # NOTE: The mock is needed as it is not clear how to trigger this exception.
    @patch(
        "cli.lab.get_taxonomy_diff",
        MagicMock(
            side_effect=GenerateException("Make sure the yaml is formatted correctly")
        ),
    )
    def test_list_generator_error(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(
                lab.list,
                [
                    "--taxonomy-base",
                    TAXONOMY_BASE,
                    "--taxonomy-path",
                    self.taxonomy.root,
                ],
            )
            self.assertIn(
                "Generating dataset failed with the following error", result.output
            )
            self.assertEqual(result.exit_code, 0)

    def test_list_invalid_base(self):
        taxonomy_base = "main"
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(
                lab.list,
                [
                    "--taxonomy-base",
                    taxonomy_base,
                    "--taxonomy-path",
                    self.taxonomy.root,
                ],
            )
            self.assertIsInstance(result.exception, SystemExit)
            self.assertIn(
                f'Couldn\'t find the taxonomy base branch "{taxonomy_base}" '
                "from the current HEAD",
                result.output,
            )
            self.assertEqual(result.exit_code, 1)

    def test_list_invalid_path(self):
        taxonomy_path = "/path/to/taxonomy"
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(
                lab.list,
                [
                    "--taxonomy-base",
                    TAXONOMY_BASE,
                    "--taxonomy-path",
                    taxonomy_path,
                ],
            )
            self.assertIsInstance(result.exception, NoSuchPathError)
            self.assertEqual(result.exit_code, 1)
