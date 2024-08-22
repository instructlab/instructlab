# SPDX-License-Identifier: Apache-2.0
# pylint: disable=duplicate-code

# Standard
from pathlib import Path
import re

# Third Party
from click.testing import CliRunner
import pytest

# First Party
from instructlab import lab

TAXONOMY_BASE = "main"

TEST_CUSTOM_YAML_RULES = b"""empty-lines: relaxed

rules:
  line-length:
    max: 180
"""


class TestLabDiff:
    """Test collection for `ilab taxonomy diff` command."""

    @pytest.fixture(autouse=True)
    def _init_taxonomy(self, taxonomy_dir):
        self.taxonomy = taxonomy_dir

    def test_diff(self, cli_runner: CliRunner):
        untracked_file = "compositional_skills/new/qna.yaml"
        tracked_file = "compositional_skills/tracked/qna.yaml"
        self.taxonomy.create_untracked(untracked_file)
        self.taxonomy.add_tracked(tracked_file)
        result = cli_runner.invoke(
            lab.ilab,
            [
                "--config=DEFAULT",
                "taxonomy",
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

    def test_diff_empty_base(self, cli_runner: CliRunner):
        taxonomy_base = "empty"
        untracked_file = "compositional_skills/new/qna.yaml"
        tracked_file = "compositional_skills/tracked/qna.yaml"
        self.taxonomy.create_untracked(untracked_file)
        self.taxonomy.add_tracked(tracked_file)
        result = cli_runner.invoke(
            lab.ilab,
            [
                "--config=DEFAULT",
                "taxonomy",
                "diff",
                "--taxonomy-base",
                taxonomy_base,
                "--taxonomy-path",
                self.taxonomy.root,
            ],
        )
        assert untracked_file in result.output
        assert tracked_file in result.output
        assert result.exit_code == 0

    def test_diff_rm_tracked(self, cli_runner: CliRunner):
        tracked_file = "compositional_skills/tracked/qna.yaml"
        self.taxonomy.add_tracked(tracked_file)
        self.taxonomy.remove_file(tracked_file)
        result = cli_runner.invoke(
            lab.ilab,
            [
                "--config=DEFAULT",
                "taxonomy",
                "diff",
                "--taxonomy-base",
                TAXONOMY_BASE,
                "--taxonomy-path",
                self.taxonomy.root,
            ],
        )
        assert tracked_file not in result.output
        assert result.exit_code == 0

    def test_diff_invalid_ext(self, cli_runner: CliRunner):
        untracked_file = "compositional_skills/writing/new/qna.txt"
        self.taxonomy.create_untracked(untracked_file)
        result = cli_runner.invoke(
            lab.ilab,
            [
                "--config=DEFAULT",
                "taxonomy",
                "diff",
                "--taxonomy-base",
                TAXONOMY_BASE,
                "--taxonomy-path",
                self.taxonomy.root,
            ],
        )

        assert untracked_file in self.taxonomy.untracked_files
        # Invalid extension is silently filtered out
        assert untracked_file not in result.output
        assert result.exit_code == 0

    def test_diff_yml_ext(self, cli_runner: CliRunner):
        untracked_file = "compositional_skills/writing/new/qna.yml"
        self.taxonomy.create_untracked(untracked_file)
        result = cli_runner.invoke(
            lab.ilab,
            [
                "--config=DEFAULT",
                "taxonomy",
                "diff",
                "--taxonomy-base",
                TAXONOMY_BASE,
                "--taxonomy-path",
                self.taxonomy.root,
            ],
        )

        assert untracked_file in self.taxonomy.untracked_files
        # Invalid extension is filtered out but there is a warning
        assert re.search(
            f"{re.escape(untracked_file)}.*must be named 'qna\\.yaml'",
            result.output,
            re.MULTILINE,
        )
        assert not re.search(
            f"^{re.escape(untracked_file)}", result.output, re.MULTILINE
        )
        assert result.exit_code == 0

    def test_diff_YAML_ext(self, cli_runner: CliRunner):
        untracked_file = "compositional_skills/writing/new/qna.YAML"
        self.taxonomy.create_untracked(untracked_file)
        result = cli_runner.invoke(
            lab.ilab,
            [
                "--config=DEFAULT",
                "taxonomy",
                "diff",
                "--taxonomy-base",
                TAXONOMY_BASE,
                "--taxonomy-path",
                self.taxonomy.root,
            ],
        )

        assert untracked_file in self.taxonomy.untracked_files
        # Invalid extension is filtered out but there is a warning
        assert re.search(
            f"{re.escape(untracked_file)}.*must be named 'qna\\.yaml'",
            result.output,
            re.MULTILINE,
        )
        assert not re.search(
            f"^{re.escape(untracked_file)}", result.output, re.MULTILINE
        )
        assert result.exit_code == 0

    def test_diff_invalid_base(self, cli_runner: CliRunner):
        taxonomy_base = "invalid"
        result = cli_runner.invoke(
            lab.ilab,
            [
                "--config=DEFAULT",
                "taxonomy",
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

    def test_diff_invalid_path(self, cli_runner: CliRunner):
        taxonomy_path = "/path/to/taxonomy"
        result = cli_runner.invoke(
            lab.ilab,
            [
                "--config=DEFAULT",
                "taxonomy",
                "diff",
                "--taxonomy-base",
                TAXONOMY_BASE,
                "--taxonomy-path",
                taxonomy_path,
            ],
        )
        assert f"{taxonomy_path}" in result.output
        assert result.exit_code == 1

    def test_diff_valid_yaml(self, cli_runner: CliRunner, testdata_path: Path):
        qna = testdata_path.joinpath("skill_valid_answer.yaml").read_bytes()
        valid_yaml_file = "compositional_skills/valid/qna.yaml"
        self.taxonomy.create_untracked(valid_yaml_file, qna)
        result = cli_runner.invoke(
            lab.ilab,
            [
                "--config=DEFAULT",
                "taxonomy",
                "diff",
                "--taxonomy-base",
                TAXONOMY_BASE,
                "--taxonomy-path",
                self.taxonomy.root,
            ],
        )
        assert f"Taxonomy in {self.taxonomy.root} is valid :)" in result.output
        assert result.exit_code == 0

    def test_diff_valid_yaml_file(self, cli_runner: CliRunner, testdata_path):
        qna = testdata_path.joinpath("skill_valid_answer.yaml").read_bytes()
        valid_yaml_file = "compositional_skills/valid/qna.yaml"
        file_path = self.taxonomy.create_untracked(valid_yaml_file, qna)
        result = cli_runner.invoke(
            lab.ilab,
            [
                "--config=DEFAULT",
                "taxonomy",
                "diff",
                "--taxonomy-base",
                TAXONOMY_BASE,
                "--taxonomy-path",
                file_path,
            ],
        )
        assert f"Taxonomy in {file_path} is valid :)" in result.output
        assert result.exit_code == 0

    def test_diff_valid_yaml_quiet(self, cli_runner: CliRunner, testdata_path):
        qna = testdata_path.joinpath("skill_valid_answer.yaml").read_bytes()
        valid_yaml_file = "compositional_skills/valid/qna.yaml"
        self.taxonomy.create_untracked(valid_yaml_file, qna)
        result = cli_runner.invoke(
            lab.ilab,
            [
                "--config=DEFAULT",
                "taxonomy",
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

    def test_diff_valid_yaml_quiet_file(self, cli_runner: CliRunner, testdata_path):
        qna = testdata_path.joinpath("skill_valid_answer.yaml").read_bytes()
        valid_yaml_file = "compositional_skills/valid/qna.yaml"
        file_path = self.taxonomy.create_untracked(valid_yaml_file, qna)
        result = cli_runner.invoke(
            lab.ilab,
            [
                "--config=DEFAULT",
                "taxonomy",
                "diff",
                "--taxonomy-base",
                TAXONOMY_BASE,
                "--taxonomy-path",
                file_path,
                "--quiet",
            ],
        )
        assert result.output == ""
        assert result.exit_code == 0

    def test_diff_invalid_yaml_content(self, cli_runner: CliRunner, testdata_path):
        qna = testdata_path.joinpath("invalid_yaml.yaml").read_bytes()
        self.taxonomy.create_untracked("compositional_skills/invalid/qna.yaml", qna)
        result = cli_runner.invoke(
            lab.ilab,
            [
                "--config=DEFAULT",
                "taxonomy",
                "diff",
                "--taxonomy-base",
                TAXONOMY_BASE,
                "--taxonomy-path",
                self.taxonomy.root,
            ],
        )
        assert "Reading taxonomy failed" in result.output
        assert result.exit_code == 1

    def test_diff_invalid_yaml_content_quiet(
        self, cli_runner: CliRunner, testdata_path
    ):
        qna = testdata_path.joinpath("invalid_yaml.yaml").read_bytes()
        self.taxonomy.create_untracked("compositional_skills/invalid/qna.yaml", qna)
        result = cli_runner.invoke(
            lab.ilab,
            [
                "--config=DEFAULT",
                "taxonomy",
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

    def test_diff_custom_yaml(self, cli_runner: CliRunner, testdata_path: Path):
        qna = testdata_path.joinpath("invalid_yaml.yaml").read_bytes()
        custom_rules_file = Path("custom_rules.yaml")
        custom_rules_file.write_bytes(TEST_CUSTOM_YAML_RULES)
        invalid_yaml_file = "compositional_skills/invalid/qna.yaml"
        self.taxonomy.create_untracked(invalid_yaml_file, qna)
        result = cli_runner.invoke(
            lab.ilab,
            [
                "--config=DEFAULT",
                "taxonomy",
                "diff",
                "--taxonomy-base",
                TAXONOMY_BASE,
                "--taxonomy-path",
                self.taxonomy.root,
                "--yaml-rules",
                str(custom_rules_file),
                "--quiet",
            ],
        )
        # custom yaml rules mean "invalid" yaml file should pass
        assert result.output == ""
        assert result.exit_code == 0
        Path.unlink(custom_rules_file)

    def test_diff_failing_schema_yaml(self, cli_runner: CliRunner, testdata_path: Path):
        qna = testdata_path.joinpath("skill_incomplete.yaml").read_bytes()
        failing_yaml_file = "compositional_skills/failing/qna.yaml"
        self.taxonomy.create_untracked(failing_yaml_file, qna)
        result = cli_runner.invoke(
            lab.ilab,
            [
                "--config=DEFAULT",
                "taxonomy",
                "diff",
                "--taxonomy-base",
                TAXONOMY_BASE,
                "--taxonomy-path",
                self.taxonomy.root,
            ],
        )
        assert "Reading taxonomy failed" in result.output
        assert result.exit_code == 1

    def test_diff_unsupported_knowledge_yaml(
        self, cli_runner: CliRunner, testdata_path: Path
    ):
        qna = testdata_path.joinpath("knowledge_unsupported.yaml").read_bytes()
        failing_yaml_file = "knowledge/unsupported/qna.yaml"
        self.taxonomy.create_untracked(failing_yaml_file, qna)
        result = cli_runner.invoke(
            lab.ilab,
            [
                "--config=DEFAULT",
                "taxonomy",
                "diff",
                "--taxonomy-base",
                TAXONOMY_BASE,
                "--taxonomy-path",
                self.taxonomy.root,
            ],
        )
        assert "Reading taxonomy failed" in result.output
        assert result.exit_code == 1

    def test_diff_invalid_yaml_filename(self, cli_runner: CliRunner, testdata_path):
        qna = testdata_path.joinpath("invalid_yaml.yaml").read_bytes()
        self.taxonomy.create_untracked(
            "compositional_skills/invalid/qna_invalid.yaml", qna
        )
        result = cli_runner.invoke(
            lab.ilab,
            [
                "--config=DEFAULT",
                "taxonomy",
                "diff",
                "--taxonomy-base",
                TAXONOMY_BASE,
                "--taxonomy-path",
                self.taxonomy.root,
            ],
        )
        assert f"Taxonomy in {self.taxonomy.root} is valid :)" in result.output
        assert result.exit_code == 0

    def test_diff_invalid_yaml_filename_quiet(
        self, cli_runner: CliRunner, testdata_path
    ):
        qna = testdata_path.joinpath("invalid_yaml.yaml").read_bytes()
        self.taxonomy.create_untracked(
            "compositional_skills/invalid/qna_invalid.yaml", qna
        )
        result = cli_runner.invoke(
            lab.ilab,
            [
                "--config=DEFAULT",
                "taxonomy",
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
