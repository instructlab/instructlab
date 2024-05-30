# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
from unittest.mock import patch
import os
import platform
import sys

# Third Party
from click.testing import CliRunner
import pytest

# First Party
from instructlab import lab
from instructlab.train import linux_train

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
    for f_path in ["/train_1.jsonl", "/test_1.jsonl"]:
        with open(INPUT_DIR + f_path, "w", encoding=ENCODING):
            pass


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
class TestLabTrain:
    """Test collection for `ilab train` command."""

    @patch("instructlab.lab.utils.is_macos_with_m_chip", return_value=True)
    @patch("instructlab.mlx_explore.gguf_convert_to_mlx.load")
    @patch("instructlab.train.lora_mlx.make_data.make_data")
    @patch("instructlab.train.lora_mlx.convert.convert_between_mlx_and_pytorch")
    @patch("instructlab.train.lora_mlx.lora.load_and_train")
    def test_train_mac(
        self,
        load_and_train_mock,
        convert_between_mlx_and_pytorch_mock,
        make_data_mock,
        load_mock,
        is_macos_with_m_chip_mock,
    ):
        runner = CliRunner()
        with runner.isolated_filesystem():
            setup_input_dir()
            result = runner.invoke(
                lab.cli, ["--config=DEFAULT", "train", "--input-dir", INPUT_DIR]
            )
            assert result.exit_code == 0
            load_mock.assert_not_called()
            load_and_train_mock.assert_called_once()
            assert load_and_train_mock.call_args[1]["model"] is not None
            assert load_and_train_mock.call_args[1]["train"]
            assert load_and_train_mock.call_args[1]["data"] == "./taxonomy_data"
            assert load_and_train_mock.call_args[1]["adapter_file"] is not None
            assert load_and_train_mock.call_args[1]["iters"] == 100
            assert load_and_train_mock.call_args[1]["save_every"] == 10
            assert load_and_train_mock.call_args[1]["steps_per_eval"] == 10
            assert len(load_and_train_mock.call_args[1]) == 7
            convert_between_mlx_and_pytorch_mock.assert_called_once()
            assert (
                convert_between_mlx_and_pytorch_mock.call_args[1]["hf_path"] is not None
            )
            assert (
                convert_between_mlx_and_pytorch_mock.call_args[1]["mlx_path"]
                is not None
            )
            assert convert_between_mlx_and_pytorch_mock.call_args[1]["quantize"]
            assert not convert_between_mlx_and_pytorch_mock.call_args[1]["local"]
            assert len(convert_between_mlx_and_pytorch_mock.call_args[1]) == 4
            make_data_mock.assert_called_once()
            assert make_data_mock.call_args[1]["data_dir"] == "./taxonomy_data"
            assert len(make_data_mock.call_args[1]) == 1
            is_macos_with_m_chip_mock.assert_called_once()

    @patch("instructlab.lab.utils.is_macos_with_m_chip", return_value=True)
    @patch("instructlab.mlx_explore.gguf_convert_to_mlx.load")
    @patch("instructlab.train.lora_mlx.make_data.make_data")
    @patch("instructlab.train.lora_mlx.convert.convert_between_mlx_and_pytorch")
    @patch("instructlab.train.lora_mlx.lora.load_and_train")
    def test_skip_quantize(
        self,
        load_and_train_mock,
        convert_between_mlx_and_pytorch_mock,
        make_data_mock,
        load_mock,
        is_macos_with_m_chip_mock,
    ):
        runner = CliRunner()
        with runner.isolated_filesystem():
            setup_input_dir()
            result = runner.invoke(
                lab.cli,
                [
                    "--config=DEFAULT",
                    "train",
                    "--input-dir",
                    INPUT_DIR,
                    "--skip-quantize",
                ],
            )
            assert result.exit_code == 0
            load_mock.assert_not_called()
            load_and_train_mock.assert_called_once()
            convert_between_mlx_and_pytorch_mock.assert_called_once()
            assert (
                convert_between_mlx_and_pytorch_mock.call_args[1]["quantize"] is False
            )
            make_data_mock.assert_called_once()
            is_macos_with_m_chip_mock.assert_called_once()

    def test_input_error(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(
                lab.cli, ["--config=DEFAULT", "train", "--input-dir", "invalid"]
            )
            assert result.exception is not None
            assert "Could not read directory: invalid" in result.output
            assert result.exit_code == 1

    def test_invalid_taxonomy(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            os.mkdir(INPUT_DIR)  # Leave out the test and train files
            result = runner.invoke(
                lab.cli, ["--config=DEFAULT", "train", "--input-dir", INPUT_DIR]
            )
            assert result.exception is not None
            assert (
                f"{INPUT_DIR} does not contain training or test files, did you run `ilab generate`?"
                in result.output
            )
            assert result.exit_code == 1

    def test_invalid_data_dir(self):
        # The error comes from make_data itself so it's only really useful to test on a mac
        if is_arm_mac():
            runner = CliRunner()
            with runner.isolated_filesystem():
                os.mkdir(INPUT_DIR)  # Leave out the test and train files
                result = runner.invoke(
                    lab.cli,
                    [
                        "--config=DEFAULT",
                        "train",
                        "--data-dir",
                        "invalid",
                        "--input-dir",
                        INPUT_DIR,
                    ],
                )
                assert result.exception is not None
                assert "Could not read from data directory" in result.output
                assert result.exit_code == 1

    @patch("instructlab.lab.utils.is_macos_with_m_chip", return_value=True)
    @patch(
        "instructlab.train.lora_mlx.make_data.make_data",
        side_effect=FileNotFoundError(),
    )
    def test_invalid_data_dir_synthetic(
        self, make_data_mock, is_macos_with_m_chip_mock
    ):
        runner = CliRunner()
        with runner.isolated_filesystem():
            os.mkdir(INPUT_DIR)  # Leave out the test and train files
            result = runner.invoke(
                lab.cli,
                [
                    "--config=DEFAULT",
                    "train",
                    "--data-dir",
                    "invalid",
                    "--input-dir",
                    INPUT_DIR,
                ],
            )
            make_data_mock.assert_called_once()
            assert result.exception is not None
            assert "Could not read from data directory" in result.output
            assert result.exit_code == 1
            is_macos_with_m_chip_mock.assert_called_once()

    @patch("instructlab.lab.utils.is_macos_with_m_chip", return_value=True)
    @patch("instructlab.mlx_explore.gguf_convert_to_mlx.load")
    @patch("instructlab.train.lora_mlx.make_data.make_data")
    @patch("instructlab.train.lora_mlx.convert.convert_between_mlx_and_pytorch")
    @patch("instructlab.train.lora_mlx.lora.load_and_train")
    def test_skip_preprocessing(
        self,
        load_and_train_mock,
        convert_between_mlx_and_pytorch_mock,
        make_data_mock,
        load_mock,
        is_macos_with_m_chip_mock,
    ):
        runner = CliRunner()
        with runner.isolated_filesystem():
            setup_input_dir()
            result = runner.invoke(
                lab.cli,
                [
                    "--config=DEFAULT",
                    "train",
                    "--input-dir",
                    INPUT_DIR,
                    "--skip-preprocessing",
                ],
            )
            assert result.exit_code == 0
            load_mock.assert_not_called()
            load_and_train_mock.assert_called_once()
            convert_between_mlx_and_pytorch_mock.assert_called_once()
            make_data_mock.assert_not_called()
            is_macos_with_m_chip_mock.assert_called_once()

    @patch("instructlab.lab.utils.is_macos_with_m_chip", return_value=True)
    @patch("instructlab.mlx_explore.utils.fetch_tokenizer_from_hub")
    @patch("instructlab.mlx_explore.gguf_convert_to_mlx.load")
    @patch("instructlab.train.lora_mlx.make_data.make_data")
    @patch("instructlab.train.lora_mlx.convert.convert_between_mlx_and_pytorch")
    @patch("instructlab.train.lora_mlx.lora.load_and_train")
    def test_load(
        self,
        load_and_train_mock,
        convert_between_mlx_and_pytorch_mock,
        make_data_mock,
        load_mock,
        fetch_tokenizer_from_hub_mock,
        is_macos_with_m_chip_mock,
    ):
        runner = CliRunner()
        with runner.isolated_filesystem():
            setup_input_dir()
            setup_load()
            result = runner.invoke(
                lab.train,
                [
                    "--input-dir",
                    INPUT_DIR,
                    "--tokenizer-dir",
                    "tokenizer",
                    "--gguf-model-path",
                    "gguf_model",
                    "--model-dir",
                    MODEL_DIR,
                ],
            )
            assert result.exit_code == 0
            load_mock.assert_called_once()
            load_and_train_mock.assert_called_once()
            convert_between_mlx_and_pytorch_mock.assert_not_called()
            make_data_mock.assert_called_once()
            fetch_tokenizer_from_hub_mock.assert_called_once()
            assert fetch_tokenizer_from_hub_mock.call_args[0][0] == "tokenizer"
            assert fetch_tokenizer_from_hub_mock.call_args[0][1] == "tokenizer"
            assert len(fetch_tokenizer_from_hub_mock.call_args[0]) == 2
            is_macos_with_m_chip_mock.assert_called_once()

    @patch("instructlab.lab.utils.is_macos_with_m_chip", return_value=True)
    @patch("instructlab.mlx_explore.utils.fetch_tokenizer_from_hub")
    @patch("instructlab.mlx_explore.gguf_convert_to_mlx.load")
    @patch("instructlab.train.lora_mlx.make_data.make_data")
    @patch("instructlab.train.lora_mlx.convert.convert_between_mlx_and_pytorch")
    @patch("instructlab.train.lora_mlx.lora.load_and_train")
    def test_load_local(
        self,
        load_and_train_mock,
        convert_between_mlx_and_pytorch_mock,
        make_data_mock,
        load_mock,
        fetch_tokenizer_from_hub_mock,
        is_macos_with_m_chip_mock,
    ):
        runner = CliRunner()
        with runner.isolated_filesystem():
            setup_input_dir()
            setup_load()
            result = runner.invoke(
                lab.train,
                [
                    "--input-dir",
                    INPUT_DIR,
                    "--tokenizer-dir",
                    "tokenizer",
                    "--gguf-model-path",
                    "gguf_model",
                    "--model-dir",
                    MODEL_DIR,
                    "--local",
                ],
            )
            assert result.exit_code == 0
            load_mock.assert_called_once()
            load_and_train_mock.assert_called_once()
            convert_between_mlx_and_pytorch_mock.assert_not_called()
            make_data_mock.assert_called_once()
            fetch_tokenizer_from_hub_mock.assert_not_called()
            is_macos_with_m_chip_mock.assert_called_once()

    @patch("instructlab.lab.utils.is_macos_with_m_chip", return_value=False)
    @patch.object(linux_train, "linux_train")
    @patch(
        "instructlab.llamacpp.llamacpp_convert_to_gguf.convert_llama_to_gguf",
        side_effect=mock_convert_llama_to_gguf,
    )
    def test_train_linux(
        self,
        convert_llama_to_gguf_mock,
        linux_train_mock,
        is_macos_with_m_chip_mock,
    ):
        runner = CliRunner()
        with runner.isolated_filesystem():
            setup_input_dir()
            setup_linux_dir()
            result = runner.invoke(
                lab.cli, ["--config=DEFAULT", "train", "--input-dir", INPUT_DIR]
            )
            assert result.exit_code == 0
            convert_llama_to_gguf_mock.assert_called_once()
            assert convert_llama_to_gguf_mock.call_args[1]["model"] == Path(
                "./training_results/final"
            )
            assert convert_llama_to_gguf_mock.call_args[1]["pad_vocab"] is True
            assert len(convert_llama_to_gguf_mock.call_args[1]) == 2
            linux_train_mock.assert_called_once()
            print(linux_train_mock.call_args[1])
            assert linux_train_mock.call_args[1]["train_file"] == Path(
                "test_generated/train_1.jsonl"
            )
            assert linux_train_mock.call_args[1]["test_file"] == Path(
                "test_generated/test_1.jsonl"
            )
            assert linux_train_mock.call_args[1]["num_epochs"] == 1
            assert linux_train_mock.call_args[1]["device"] is not None
            assert not linux_train_mock.call_args[1]["four_bit_quant"]
            assert len(linux_train_mock.call_args[1]) == 7
            is_macos_with_m_chip_mock.assert_called_once()
            assert not os.path.isfile(LINUX_GGUF_FILE)

    @patch("instructlab.lab.utils.is_macos_with_m_chip", return_value=False)
    @patch("instructlab.train.linux_train.linux_train")
    @patch(
        "instructlab.llamacpp.llamacpp_convert_to_gguf.convert_llama_to_gguf",
        side_effect=mock_convert_llama_to_gguf,
    )
    def test_num_epochs(
        self, convert_llama_to_gguf_mock, linux_train_mock, is_macos_with_m_chip_mock
    ):
        runner = CliRunner()
        with runner.isolated_filesystem():
            setup_input_dir()
            setup_linux_dir()
            result = runner.invoke(
                lab.cli,
                [
                    "--config=DEFAULT",
                    "train",
                    "--input-dir",
                    INPUT_DIR,
                    "--num-epochs",
                    "2",
                ],
            )
            assert result.exit_code == 0
            convert_llama_to_gguf_mock.assert_called_once()
            linux_train_mock.assert_called_once()
            assert linux_train_mock.call_args[1]["num_epochs"] == 2
            is_macos_with_m_chip_mock.assert_called_once()
            assert not os.path.isfile(LINUX_GGUF_FILE)

            # Test with invalid num_epochs
            result = runner.invoke(
                lab.cli,
                [
                    "--config=DEFAULT",
                    "train",
                    "--input-dir",
                    INPUT_DIR,
                    "--num-epochs",
                    "two",
                ],
            )
            assert result.exception is not None
            assert result.exit_code == 2
            assert "'two' is not a valid integer" in result.output
