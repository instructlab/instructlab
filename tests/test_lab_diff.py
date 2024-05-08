# SPDX-License-Identifier: Apache-2.0
# pylint: disable=duplicate-code

# Standard
from pathlib import Path

# Third Party
from click.testing import CliRunner
import pytest

# First Party
from instructlab import lab

TAXONOMY_BASE = "main"

TEST_CUSTOM_YAML_RULES = b"""extends: relaxed

rules:
  line-length:
    max: 180
"""


class TestLabDiff:
    """Test collection for `ilab diff` command."""

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
                lab.cli,
                [
                    "--config=DEFAULT",
                    "diff",
                    "--taxonomy-base",
                    TAXONOMY_BASE,
                    "--taxonomy-path",
                    self.taxonomy.root,
                ],
            )
            assert untracked_file in result.output
            assert tracked_file not in result.output
            assert result.exit_code == 0

    def test_diff_rm_tracked(self):
        tracked_file = "compositional_skills/tracked/qna.yaml"
        self.taxonomy.add_tracked(tracked_file)
        self.taxonomy.remove_file(tracked_file)
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(
                lab.cli,
                [
                    "--config=DEFAULT",
                    "diff",
                    "--taxonomy-base",
                    TAXONOMY_BASE,
                    "--taxonomy-path",
                    self.taxonomy.root,
                ],
            )
            assert tracked_file not in result.output
            assert result.exit_code == 0

    def test_diff_invalid_ext(self):
        untracked_file = "compositional_skills/writing/new/qna.YAML"
        self.taxonomy.create_untracked(untracked_file)
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(
                lab.cli,
                [
                    "--config=DEFAULT",
                    "diff",
                    "--taxonomy-base",
                    TAXONOMY_BASE,
                    "--taxonomy-path",
                    self.taxonomy.root,
                ],
            )
            assert self.taxonomy.untracked_files == [untracked_file]
            # Invalid extension is silently filtered out
            assert untracked_file not in result.output
            assert result.exit_code == 0

    def test_diff_invalid_base(self):
        taxonomy_base = "invalid"
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(
                lab.cli,
                [
                    "--config=DEFAULT",
                    "diff",
                    "--taxonomy-base",
                    taxonomy_base,
                    "--taxonomy-path",
                    self.taxonomy.root,
                ],
            )
            assert (
                f'Couldn\'t find the taxonomy git ref "{taxonomy_base}" '
                "from the current HEAD" in result.output
            )
            assert result.exit_code == 1

    def test_diff_invalid_path(self):
        taxonomy_path = "/path/to/taxonomy"
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(
                lab.cli,
                [
                    "--config=DEFAULT",
                    "diff",
                    "--taxonomy-base",
                    TAXONOMY_BASE,
                    "--taxonomy-path",
                    taxonomy_path,
                ],
            )
            assert f"{taxonomy_path}" in result.output
            assert result.exit_code == 1

    def test_diff_valid_yaml(self):
        with open("tests/testdata/skill_valid_answer.yaml", "rb") as qnafile:
            valid_yaml_file = "compositional_skills/qna_valid.yaml"
            self.taxonomy.create_untracked(valid_yaml_file, qnafile.read())
            runner = CliRunner()
            result = runner.invoke(
                lab.cli,
                [
                    "--config=DEFAULT",
                    "diff",
                    "--taxonomy-base",
                    TAXONOMY_BASE,
                    "--taxonomy-path",
                    self.taxonomy.root,
                ],
            )
            assert f"Taxonomy in /{self.taxonomy.root}/ is valid :)" in result.output
            assert result.exit_code == 0

    def test_diff_valid_yaml_quiet(self):
        with open("tests/testdata/skill_valid_answer.yaml", "rb") as qnafile:
            valid_yaml_file = "compositional_skills/qna_valid.yaml"
            self.taxonomy.create_untracked(valid_yaml_file, qnafile.read())
            runner = CliRunner()
            result = runner.invoke(
                lab.cli,
                [
                    "--config=DEFAULT",
                    "diff",
                    "--taxonomy-base",
                    TAXONOMY_BASE,
                    "--taxonomy-path",
                    self.taxonomy.root,
                    "--quiet",
                ],
            )
            assert result.output == ""
            assert result.exit_code == 0

    def test_diff_invalid_yaml(self):
        with open("tests/testdata/invalid_yaml.yaml", "rb") as qnafile:
            self.taxonomy.create_untracked(
                "compositional_skills/qna_invalid.yaml", qnafile.read()
            )
            runner = CliRunner()
            result = runner.invoke(
                lab.cli,
                [
                    "--config=DEFAULT",
                    "diff",
                    "--taxonomy-base",
                    TAXONOMY_BASE,
                    "--taxonomy-path",
                    self.taxonomy.root,
                ],
            )
            assert "Reading taxonomy failed" in result.output
            assert result.exit_code == 1

    def test_diff_invalid_yaml_quiet(self):
        with open("tests/testdata/invalid_yaml.yaml", "rb") as qnafile:
            self.taxonomy.create_untracked(
                "compositional_skills/qna_invalid.yaml", qnafile.read()
            )
            runner = CliRunner()
            result = runner.invoke(
                lab.cli,
                [
                    "--config=DEFAULT",
                    "diff",
                    "--taxonomy-base",
                    TAXONOMY_BASE,
                    "--taxonomy-path",
                    self.taxonomy.root,
                    "--quiet",
                ],
            )
            assert result.exception is not None
            assert result.exit_code == 1

    def test_diff_custom_yaml(self):
        with open("tests/testdata/invalid_yaml.yaml", "rb") as qnafile:
            custom_rules_file = Path("custom_rules.yaml")
            custom_rules_file.write_bytes(TEST_CUSTOM_YAML_RULES)
            invalid_yaml_file = "compositional_skills/qna_invalid.yaml"
            self.taxonomy.create_untracked(invalid_yaml_file, qnafile.read())
            runner = CliRunner()
            result = runner.invoke(
                lab.cli,
                [
                    "--config=DEFAULT",
                    "diff",
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
            assert result.output == ""
            assert result.exit_code == 0
            Path.unlink(custom_rules_file)

    def test_diff_failing_schema_yaml(self):
        with open("tests/testdata/skill_incomplete.yaml", "rb") as qnafile:
            failing_yaml_file = "compositional_skills/failing/qna.yaml"
            self.taxonomy.create_untracked(failing_yaml_file, qnafile.read())
            runner = CliRunner()
            result = runner.invoke(
                lab.cli,
                [
                    "--config=DEFAULT",
                    "diff",
                    "--taxonomy-base",
                    TAXONOMY_BASE,
                    "--taxonomy-path",
                    self.taxonomy.root,
                ],
            )
            assert "Reading taxonomy failed" in result.output
            assert result.exit_code == 1
