# Standard
from unittest.mock import patch
import fnmatch
import logging
import os
import pathlib
import unittest

# Third Party
from click.testing import CliRunner

# First Party
from cli import lab
from cli.generator.generate_data import generate_data
from cli.generator.utils import GenerateException

# Local
from .taxonomy import MockTaxonomy

test_skill_valid_answer = b"""created_by: test-bot
seed_examples:
- answer: Yes, it is.
  question: Is this for a test?
- answer: Yes I am very sure.
  question: Are you sure it's for a test?
task_description: for testing
"""


test_skill_invalid_answer = b"""created_by: test-bot
seed_examples:
- answer: Yes
  question: Is this for a test?
task_description: for testing
"""

generate_data_return_value = [
    {
        "instruction": "3. Tell me a pun about water.",
        "input": "",
        "output": "Why did the scarecrow win an award?\nBecause he was outstanding in his field!",
        "taxonomy_path": "compositional_skills->writing->freeform->jokes->puns-copy->general",
        "task_description": "to teach a large language model to come up with puns",
        "document": None,
    },
    {
        "instruction": "4. Give me a pun about books.",
        "input": "",
        "output": "Why don't books ever get lost on the shelf?\nBecause they are always on the cover!",
        "taxonomy_path": "compositional_skills->writing->freeform->jokes->puns-copy->general",
        "task_description": "to teach a large language model to come up with puns",
        "document": None,
    },
]


class TestLabGenerate(unittest.TestCase):
    """Test collection for `ilab generate` command."""

    @patch(
        "cli.generator.generate_data.generate_data",
        side_effect=GenerateException("Connection Error"),
    )
    def test_generate_exception_error(self, generate_data_mock):
        runner = CliRunner()
        with runner.isolated_filesystem():
            mt = MockTaxonomy(pathlib.Path("taxonomy"))
            result = runner.invoke(
                lab.generate,
                [
                    "--taxonomy-base",
                    "main",
                    "--taxonomy-path",
                    mt.root,
                    "--endpoint-url",
                    "localhost:8000",
                ],
            )
            self.assertEqual(
                result.exit_code, 1, "command finished with an unexpected exit code"
            )
            generate_data_mock.assert_called_once()
            self.assertIn(
                "Generating dataset failed with the following error: Connection Error",
                result.output,
            )

    def test_taxonomy_not_found(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(
                lab.generate,
                [
                    "--endpoint-url",
                    "localhost:8000",
                ],
            )
            self.assertEqual(
                result.exit_code, 1, "command finished with an unexpected exit code"
            )
            self.assertIn(
                "Error: taxonomy (taxonomy) does not exist",
                result.output,
            )

    def test_no_new_data(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            mt = MockTaxonomy(pathlib.Path("taxonomy"))
            result = runner.invoke(
                lab.generate,
                [
                    "--taxonomy-base",
                    "main",
                    "--taxonomy-path",
                    mt.root,
                    "--endpoint-url",
                    "localhost:8000",
                ],
            )
            self.assertEqual(
                result.exit_code, 1, "command finished with an unexpected exit code"
            )
            self.assertIn(
                "Nothing to generate. Exiting.",
                result.output,
            )

    def test_new_data_invalid_answer(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            mt = MockTaxonomy(pathlib.Path("taxonomy"))
            mt.create_untracked(
                "compositional_skills/tracked/qna.yaml", test_skill_invalid_answer
            )
            result = runner.invoke(
                lab.generate,
                [
                    "--taxonomy-base",
                    "main",
                    "--taxonomy-path",
                    mt.root,
                    "--endpoint-url",
                    "localhost:8000",
                ],
            )
            self.assertEqual(
                result.exit_code, 1, "command finished with an unexpected exit code"
            )
            self.assertIn(
                "Error reading seed examples: encoding without a string argument. Please make sure your answers are verbose enough",
                result.output,
            )

    @patch(
        "cli.generator.generate_data.get_instructions_from_model",
        side_effect=GenerateException(
            "There was a problem connecting to the OpenAI server."
        ),
    )
    def test_OpenAI_server_error(self, get_instructions_from_model):
        with CliRunner().isolated_filesystem():
            mt = MockTaxonomy(pathlib.Path("taxonomy"))
            mt.create_untracked(
                "compositional_skills/tracked/qna.yaml", test_skill_valid_answer
            )
            with self.assertRaises(GenerateException) as exc:
                generate_data(
                    logger=logging.getLogger("test_logger"),
                    api_base="localhost:8000",
                    api_key="",
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
            self.assertIn(
                "There was a problem connecting to the OpenAI server",
                f"{exc.exception}",
            )
            get_instructions_from_model.assert_called_once()

    @patch(
        "cli.generator.generate_data.get_instructions_from_model",
        return_value=(generate_data_return_value, 0),
    )
    def test_no_error(self, get_instructions_from_model):
        with CliRunner().isolated_filesystem():
            mt = MockTaxonomy(pathlib.Path("taxonomy"))
            mt.create_untracked(
                "compositional_skills/tracked/qna.yaml", test_skill_valid_answer
            )
            generate_data(
                logger=logging.getLogger("test_logger"),
                api_base="localhost:8000",
                api_key="",
                model_name="my-model",
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
                self.assertTrue(
                    any(fnmatch.fnmatch(f, pattern) for pattern in expected_files)
                )
