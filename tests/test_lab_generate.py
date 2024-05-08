# SPDX-License-Identifier: Apache-2.0
# pylint: disable=duplicate-code

# Standard
from unittest.mock import patch
import fnmatch
import logging
import os
import pathlib

# Third Party
from click.testing import CliRunner
import pytest

# First Party
from instructlab import lab
from instructlab.generator.generate_data import generate_data
from instructlab.generator.utils import GenerateException

# Local
from .taxonomy import MockTaxonomy
from .testdata import testdata


class TestLabGenerate:
    """Test collection for `ilab generate` command."""

    @patch(
        "instructlab.generator.generate_data.generate_data",
        side_effect=GenerateException("Connection Error"),
    )
    def test_generate_exception_error(self, generate_data_mock):
        runner = CliRunner()
        with runner.isolated_filesystem():
            mt = MockTaxonomy(pathlib.Path("taxonomy"))
            result = runner.invoke(
                lab.cli,
                [
                    "--config=DEFAULT",
                    "generate",
                    "--taxonomy-base",
                    "main",
                    "--taxonomy-path",
                    mt.root,
                    "--endpoint-url",
                    "localhost:8000",
                ],
            )
            assert (
                result.exit_code == 1
            ), "command finished with an unexpected exit code"
            generate_data_mock.assert_called_once()
            assert (
                "Generating dataset failed with the following error: Connection Error"
                in result.output
            )
            mt.teardown()

    def test_taxonomy_not_found(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(
                lab.cli,
                [
                    "--config=DEFAULT",
                    "generate",
                    "--endpoint-url",
                    "localhost:8000",
                ],
            )
            assert (
                result.exit_code == 1
            ), "command finished with an unexpected exit code"
            assert "Error: taxonomy (taxonomy) does not exist" in result.output

    def test_no_new_data(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            mt = MockTaxonomy(pathlib.Path("taxonomy"))
            result = runner.invoke(
                lab.cli,
                [
                    "--config=DEFAULT",
                    "generate",
                    "--taxonomy-base",
                    "main",
                    "--taxonomy-path",
                    mt.root,
                    "--endpoint-url",
                    "localhost:8000",
                ],
            )
            assert (
                result.exit_code == 1
            ), "command finished with an unexpected exit code"
            assert "Nothing to generate. Exiting." in result.output
            mt.teardown()

    def test_new_data_invalid_answer(self):
        runner = CliRunner()
        with open("tests/testdata/skill_invalid_answer.yaml", "rb") as qnafile:
            with runner.isolated_filesystem():
                mt = MockTaxonomy(pathlib.Path("taxonomy"))
                mt.create_untracked(
                    "compositional_skills/tracked/qna.yaml", qnafile.read()
                )
                result = runner.invoke(
                    lab.cli,
                    [
                        "--config=DEFAULT",
                        "generate",
                        "--taxonomy-base",
                        "main",
                        "--taxonomy-path",
                        mt.root,
                        "--endpoint-url",
                        "localhost:8000",
                    ],
                )
                assert (
                    result.exit_code == 1
                ), "command finished with an unexpected exit code"
                assert "taxonomy files with errors" in result.output
                mt.teardown()

    @patch(
        "instructlab.generator.generate_data.get_instructions_from_model",
        side_effect=GenerateException(
            "There was a problem connecting to the OpenAI server."
        ),
    )
    def test_open_ai_server_error(self, get_instructions_from_model):
        with open("tests/testdata/skill_valid_answer.yaml", "rb") as qnafile:
            with CliRunner().isolated_filesystem():
                mt = MockTaxonomy(pathlib.Path("taxonomy"))
                mt.create_untracked(
                    "compositional_skills/tracked/qna.yaml", qnafile.read()
                )
                with pytest.raises(GenerateException) as exc:
                    generate_data(
                        logger=logging.getLogger("test_logger"),
                        api_base="localhost:8000",
                        api_key="",
                        model_family="merlinite",
                        model_name="test-model",
                        num_cpus=10,
                        num_instructions_to_generate=100,
                        taxonomy=mt.root,
                        taxonomy_base="main",
                        output_dir="generated",
                        prompt_file_path="prompt.txt",
                        rouge_threshold=0.9,
                        console_output=True,
                        chunk_word_count=1000,
                        server_ctx_size=4096,
                        tls_insecure=False,
                    )
                assert "There was a problem connecting to the OpenAI server" in str(
                    exc.value
                )
                get_instructions_from_model.assert_called_once()
                mt.teardown()

    @patch(
        "instructlab.generator.generate_data.get_instructions_from_model",
        return_value=(testdata.generate_data_return_value, 0),
    )
    def test_generate_no_error(self, get_instructions_from_model):
        with open("tests/testdata/skill_valid_answer.yaml", "rb") as qnafile:
            with CliRunner().isolated_filesystem():
                mt = MockTaxonomy(pathlib.Path("taxonomy"))
                mt.create_untracked(
                    "compositional_skills/tracked/qna.yaml", qnafile.read()
                )
                generate_data(
                    logger=logging.getLogger("test_logger"),
                    api_base="localhost:8000",
                    api_key="",
                    model_name="my-model",
                    model_family="merlinite",
                    num_cpus=10,
                    num_instructions_to_generate=1,
                    taxonomy=mt.root,
                    taxonomy_base="main",
                    output_dir="generated",
                    prompt_file_path="prompt.txt",
                    rouge_threshold=0.9,
                    console_output=True,
                    chunk_word_count=1000,
                    server_ctx_size=4096,
                    tls_insecure=False,
                )
                get_instructions_from_model.assert_called_once()
                expected_files = [
                    "generated_my-model*.json",
                    "train_my-model*.jsonl",
                    "test_my-model*.jsonl",
                ]
                for f in os.listdir("generated"):
                    assert any(
                        fnmatch.fnmatch(f, pattern) for pattern in expected_files
                    )
                mt.teardown()

    @patch(
        "instructlab.generator.generate_data.get_instructions_from_model",
        return_value=(testdata.generate_data_return_value, 0),
    )
    @patch(
        "instructlab.generator.generate_data.read_taxonomy",
        return_value=testdata.knowledge_seed_instruction,
    )
    def test_knowledge_docs_no_error(self, read_taxonomy, get_instructions_from_model):
        with open("tests/testdata/knowledge_valid.yaml", "rb") as qnafile:
            with CliRunner().isolated_filesystem():
                mt = MockTaxonomy(pathlib.Path("taxonomy"))
                mt.create_untracked(
                    "knowledge/technical-manual/test/qna.yaml", qnafile.read()
                )
                generate_data(
                    logger=logging.getLogger("test_logger"),
                    api_base="localhost:8000",
                    api_key="",
                    model_name="my-model",
                    model_family="merlinite",
                    num_cpus=10,
                    num_instructions_to_generate=1,
                    taxonomy=mt.root,
                    taxonomy_base="main",
                    output_dir="generated",
                    prompt_file_path="prompt.txt",
                    rouge_threshold=0.9,
                    console_output=True,
                    chunk_word_count=1000,
                    server_ctx_size=4096,
                    tls_insecure=False,
                )
                get_instructions_from_model.assert_called_once()
                read_taxonomy.assert_called_once()
                expected_files = [
                    "generated_my-model*.json",
                    "train_my-model*.jsonl",
                    "test_my-model*.jsonl",
                ]
                for f in os.listdir("generated"):
                    assert any(
                        fnmatch.fnmatch(f, pattern) for pattern in expected_files
                    )
                mt.teardown()
