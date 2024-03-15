# Standard
import unittest

# Third Party
from click.testing import CliRunner
import pytest

# First Party
from cli import lab

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
            # Invalid extension is silently filtered out.
            self.assertNotIn(untracked_file, result.output)
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
            self.assertIsNone(result.exception)
            self.assertIn(
                f'Couldn\'t find the taxonomy base branch "{taxonomy_base}" '
                "from the current HEAD",
                result.output,
            )
            self.assertEqual(result.exit_code, 0)

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
            self.assertIsNone(result.exception)
            self.assertIn(f"{taxonomy_path}", result.output)
            self.assertEqual(result.exit_code, 0)
