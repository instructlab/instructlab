# Standard
from pathlib import Path
import unittest

# Third Party
from click.testing import CliRunner
import pytest

# First Party
from cli import lab

TAXONOMY_BASE = "main"
TEST_VALID_YAML = b"""created_by: nathan-weinberg
seed_examples:
- answer: "Yes I can!"
  context: "This unit test is valid!"
  question: "Could you write a unit test?"
task_description: ''
"""
TEST_INVALID_YAML = b"""created_by: nathan-weinberg
seed_examples:
- answer: "Yes I can!"
  context: "This unit test has a line with 124 characters! It is too long for the default rules but not too long for the customer rules!"
  question: "Could you write a unit test?"
task_description: ''
"""
TEST_CUSTOM_YAML_RULES = b"""extends: relaxed

rules:
  line-length:
    max: 180
"""


class TestLabDiff(unittest.TestCase):
    """Test collection for `lab list` command."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @pytest.fixture(autouse=True)
    def _init_taxonomy(self, taxonomy_dir):
        self.taxonomy = taxonomy_dir

    def test_diff(self):
        untracked_file = "compositional_skills/new/qna.yaml"
        tracked_file = "compositional_skills/tracked/qna.yaml"
        self.taxonomy.create_untracked(untracked_file)
        self.taxonomy.add_tracked(tracked_file)
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(
                lab.diff,
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

    def test_diff_rm_tracked(self):
        tracked_file = "compositional_skills/tracked/qna.yaml"
        self.taxonomy.add_tracked(tracked_file)
        self.taxonomy.remove_file(tracked_file)
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(
                lab.diff,
                [
                    "--taxonomy-base",
                    TAXONOMY_BASE,
                    "--taxonomy-path",
                    self.taxonomy.root,
                ],
            )
            self.assertNotIn(tracked_file, result.output)
            self.assertEqual(result.exit_code, 0)

    def test_diff_invalid_ext(self):
        untracked_file = "compositional_skills/writing/new/qna.YAML"
        self.taxonomy.create_untracked(untracked_file)
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(
                lab.diff,
                [
                    "--taxonomy-base",
                    TAXONOMY_BASE,
                    "--taxonomy-path",
                    self.taxonomy.root,
                ],
            )
            self.assertListEqual(self.taxonomy.untracked_files, [untracked_file])
            # Invalid extension is silently filtered out
            self.assertNotIn(untracked_file, result.output)
            self.assertEqual(result.exit_code, 0)

    def test_diff_invalid_base(self):
        taxonomy_base = "invalid"
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(
                lab.diff,
                [
                    "--taxonomy-base",
                    taxonomy_base,
                    "--taxonomy-path",
                    self.taxonomy.root,
                ],
            )
            self.assertIsNone(result.exception)
            self.assertIn(
                f'Couldn\'t find the taxonomy git ref "{taxonomy_base}" '
                "from the current HEAD",
                result.output,
            )
            self.assertEqual(result.exit_code, 0)

    def test_diff_invalid_path(self):
        taxonomy_path = "/path/to/taxonomy"
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(
                lab.diff,
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

    def test_diff_valid_yaml(self):
        valid_yaml_file = "compositional_skills/qna_valid.yaml"
        self.taxonomy.create_untracked(valid_yaml_file, TEST_VALID_YAML)
        runner = CliRunner()
        result = runner.invoke(
            lab.diff,
            [
                "--taxonomy-base",
                TAXONOMY_BASE,
                "--taxonomy-path",
                self.taxonomy.root,
                "--quiet",
            ],
        )
        self.assertIn(f"Taxonomy in {self.taxonomy.root} is valid :)", result.output)
        self.assertEqual(result.exit_code, 0)

    def test_diff_invalid_yaml(self):
        invalid_yaml_file = "compositional_skills/qna_invalid.yaml"
        self.taxonomy.create_untracked(invalid_yaml_file, TEST_INVALID_YAML)
        runner = CliRunner()
        result = runner.invoke(
            lab.diff,
            [
                "--taxonomy-base",
                TAXONOMY_BASE,
                "--taxonomy-path",
                self.taxonomy.root,
                "--quiet",
            ],
        )
        self.assertIn(
            "1 taxonomy files with errors!",
            result.output,
        )
        self.assertEqual(result.exit_code, 1)

    def test_diff_custom_yaml(self):
        custom_rules_file = Path("custom_rules.yaml")
        custom_rules_file.write_bytes(TEST_CUSTOM_YAML_RULES)
        invalid_yaml_file = "compositional_skills/qna_invalid.yaml"
        self.taxonomy.create_untracked(invalid_yaml_file, TEST_INVALID_YAML)
        runner = CliRunner()
        result = runner.invoke(
            lab.diff,
            [
                "--taxonomy-base",
                TAXONOMY_BASE,
                "--taxonomy-path",
                self.taxonomy.root,
                "--yaml-rules",
                custom_rules_file,
                "--quiet",
            ],
        )
        # custom yaml rules mean "invalid" yaml file should pass
        self.assertIn(f"Taxonomy in {self.taxonomy.root} is valid :)", result.output)
        self.assertEqual(result.exit_code, 0)
        Path.unlink(custom_rules_file)
