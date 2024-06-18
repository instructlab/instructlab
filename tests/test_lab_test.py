# SPDX-License-Identifier: Apache-2.0
# pylint: disable=duplicate-code

# Standard
from unittest.mock import patch
import os
import platform
import sys

# Third Party
from click.testing import CliRunner
import pytest

# First Party
from instructlab import lab

INPUT_DIR = "test_generated"
TRAINING_RESULTS_DIR = "training_results"
CHECKPOINT_DIR_NAME = "checkpoint-1"
MERGED_MODEL_DIR_NAME = "merged_model"
FINAL_RESULTS_DIR_NAME = "final"
CHECKPOINT_DIR = TRAINING_RESULTS_DIR + "/" + CHECKPOINT_DIR_NAME
MERGED_MODEL_DIR = TRAINING_RESULTS_DIR + "/" + MERGED_MODEL_DIR_NAME
FINAL_RESULTS_DIR = TRAINING_RESULTS_DIR + "/" + FINAL_RESULTS_DIR_NAME
LINUX_GGUF_FILE = FINAL_RESULTS_DIR + "/ggml-model-f16.gguf"
MODEL_DIR = "model"
ENCODING = "UTF-8"


def setup_input_dir():
    os.mkdir(INPUT_DIR)
    for f_path in ["/train.jsonl", "/test.jsonl"]:
        with open(INPUT_DIR + f_path, "w", encoding=ENCODING) as f:
            f.write(
                """{"system": "some context", "user": "some query", "assistant": "expected answer"}"""
            )


def setup_linux_dir():
    os.makedirs(CHECKPOINT_DIR)
    for f_path in [
        "/added_tokens.json",
        "/special_tokens_map.json",
        "/tokenizer.json",
        "/tokenizer.model",
        "/tokenizer_config.json",
    ]:
        with open(CHECKPOINT_DIR + f_path, "w", encoding=ENCODING) as f:
            f.write("{}")
    os.makedirs(MERGED_MODEL_DIR)
    for f_path in ["/config.json", "/generation_config.json", "/1.safetensors"]:
        with open(MERGED_MODEL_DIR + f_path, "w", encoding=ENCODING) as f:
            f.write("{}")


def setup_load():
    os.makedirs(MODEL_DIR)


def is_arm_mac():
    return sys.platform == "darwin" and platform.machine() == "arm64"


def mock_convert_llama_to_gguf(model, pad_vocab):
    with open(LINUX_GGUF_FILE, "w", encoding="utf-8") as fp:
        fp.write(str(model) + str(pad_vocab))
    return LINUX_GGUF_FILE


@pytest.mark.usefixtures("mock_mlx_package")
class TestLabModelTest:
    """Test collection for `ilab model test` command."""

    @patch("instructlab.utils.is_macos_with_m_chip", return_value=True)
    @patch("instructlab.train.lora_mlx.lora.load_and_train")
    def test_load(
        self,
        load_and_train_mock,
        is_macos_with_m_chip_mock,
    ):
        runner = CliRunner()
        with runner.isolated_filesystem():
            setup_input_dir()
            setup_load()
            result = runner.invoke(
                lab.ilab,
                [
                    "--config=DEFAULT",
                    "test",
                    "--data-dir",
                    os.path.join(os.getcwd(), INPUT_DIR),
                    "--max-tokens",
                    "300",
                    "--model-dir",
                    MODEL_DIR,
                    "--adapter-file",
                    "None",
                ],
            )
            assert result.exit_code == 0
            load_and_train_mock.assert_called_once()
            assert load_and_train_mock.call_args[1]["max_tokens"] == 300

            is_macos_with_m_chip_mock.assert_called_once()

            load_and_train_mock.reset_mock()
            result = runner.invoke(
                lab.ilab,
                [
                    "--config=DEFAULT",
                    "test",
                    "--data-dir",
                    os.path.join(os.getcwd(), INPUT_DIR),
                    "--model-dir",
                    MODEL_DIR,
                    "--adapter-file",
                    "None",
                ],
            )
            assert result.exit_code == 0
            load_and_train_mock.assert_called_once()
            assert load_and_train_mock.call_args[1]["max_tokens"] == 100
