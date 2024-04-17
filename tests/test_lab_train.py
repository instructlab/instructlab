# SPDX-License-Identifier: Apache-2.0

# Standard
from unittest.mock import MagicMock, patch
import os
import platform
import sys
import unittest

# Third Party
from click.testing import CliRunner

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
    os.makedirs(FINAL_RESULTS_DIR)
    with open(LINUX_GGUF_FILE, "w", encoding=ENCODING) as f:
        f.write("{}")


def setup_load():
    os.makedirs(MODEL_DIR)


def is_arm_mac():
    return sys.platform == "darwin" and platform.machine() == "arm64"


def mock_mlx(f):
    """mlx is not available on Linux"""
    mlx_modules = {
        name: MagicMock()
        for name in ["mlx", "mlx.core", "mlx.nn", "mlx.optimizers", "mlx.utils"]
    }
    return patch.dict(sys.modules, mlx_modules)(f)


class TestLabTrain(unittest.TestCase):
    """Test collection for `ilab train` command."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @mock_mlx
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
            result = runner.invoke(lab.train, ["--input-dir", INPUT_DIR])
            self.assertEqual(result.exit_code, 0)
            load_mock.assert_not_called()
            load_and_train_mock.assert_called_once()
            self.assertIsNotNone(load_and_train_mock.call_args[1]["model"])
            self.assertTrue(load_and_train_mock.call_args[1]["train"])
            self.assertEqual(
                load_and_train_mock.call_args[1]["data"], "./taxonomy_data"
            )
            self.assertIsNotNone(load_and_train_mock.call_args[1]["adapter_file"])
            self.assertEqual(load_and_train_mock.call_args[1]["iters"], 100)
            self.assertEqual(load_and_train_mock.call_args[1]["save_every"], 10)
            self.assertEqual(load_and_train_mock.call_args[1]["steps_per_eval"], 10)
            self.assertEqual(len(load_and_train_mock.call_args[1]), 7)
            convert_between_mlx_and_pytorch_mock.assert_called_once()
            self.assertIsNotNone(
                convert_between_mlx_and_pytorch_mock.call_args[1]["hf_path"]
            )
            self.assertIsNotNone(
                convert_between_mlx_and_pytorch_mock.call_args[1]["mlx_path"]
            )
            self.assertTrue(
                convert_between_mlx_and_pytorch_mock.call_args[1]["quantize"]
            )
            self.assertFalse(convert_between_mlx_and_pytorch_mock.call_args[1]["local"])
            self.assertEqual(len(convert_between_mlx_and_pytorch_mock.call_args[1]), 4)
            make_data_mock.assert_called_once()
            self.assertEqual(make_data_mock.call_args[1]["data_dir"], "./taxonomy_data")
            self.assertEqual(len(make_data_mock.call_args[1]), 1)
            is_macos_with_m_chip_mock.assert_called_once()

    @mock_mlx
    @patch("instructlab.lab.utils.is_macos_with_m_chip", return_value=True)
    @patch("instructlab.mlx_explore.gguf_convert_to_mlx.load")
    @patch("instructlab.train.lora_mlx.make_data.make_data")
    @patch("instructlab.train.lora_mlx.convert.convert_between_mlx_and_pytorch")
    @patch("instructlab.train.lora_mlx.lora.load_and_train")
    @mock_mlx
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
                lab.train, ["--input-dir", INPUT_DIR, "--skip-quantize"]
            )
            self.assertEqual(result.exit_code, 0)
            load_mock.assert_not_called()
            load_and_train_mock.assert_called_once()
            convert_between_mlx_and_pytorch_mock.assert_called_once()
            self.assertEqual(
                convert_between_mlx_and_pytorch_mock.call_args[1]["quantize"], False
            )
            make_data_mock.assert_called_once()
            is_macos_with_m_chip_mock.assert_called_once()

    def test_input_error(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(lab.train, ["--input-dir", "invalid"])
            self.assertIsNotNone(result.exception)
            self.assertIn("No such file or directory: 'invalid'", result.output)
            self.assertEqual(result.exit_code, 1)

    def test_invalid_taxonomy(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            os.mkdir(INPUT_DIR)  # Leave out the test and train files
            result = runner.invoke(lab.train, ["--input-dir", INPUT_DIR])
            self.assertIsNotNone(result.exception)
            self.assertIn(
                "Could not copy into data directory: list index out of range",
                result.output,
            )
            self.assertEqual(result.exit_code, 1)

    def test_invalid_data_dir(self):
        # The error comes from make_data itself so it's only really useful to test on a mac
        if is_arm_mac():
            runner = CliRunner()
            with runner.isolated_filesystem():
                os.mkdir(INPUT_DIR)  # Leave out the test and train files
                result = runner.invoke(
                    lab.train, ["--data-dir", "invalid", "--input-dir", INPUT_DIR]
                )
                self.assertIsNotNone(result.exception)
                self.assertIn(
                    "Could not read from data directory",
                    result.output,
                )
                self.assertEqual(result.exit_code, 1)

    @mock_mlx
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
                lab.train, ["--data-dir", "invalid", "--input-dir", INPUT_DIR]
            )
            make_data_mock.assert_called_once()
            self.assertIsNotNone(result.exception)
            self.assertIn(
                "Could not read from data directory",
                result.output,
            )
            self.assertEqual(result.exit_code, 1)
            is_macos_with_m_chip_mock.assert_called_once()

    @mock_mlx
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
                lab.train, ["--input-dir", INPUT_DIR, "--skip-preprocessing"]
            )
            self.assertEqual(result.exit_code, 0)
            load_mock.assert_not_called()
            load_and_train_mock.assert_called_once()
            convert_between_mlx_and_pytorch_mock.assert_called_once()
            make_data_mock.assert_not_called()
            is_macos_with_m_chip_mock.assert_called_once()

    @mock_mlx
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
            self.assertEqual(result.exit_code, 0)
            load_mock.assert_called_once()
            load_and_train_mock.assert_called_once()
            convert_between_mlx_and_pytorch_mock.assert_not_called()
            make_data_mock.assert_called_once()
            fetch_tokenizer_from_hub_mock.assert_called_once()
            self.assertEqual(fetch_tokenizer_from_hub_mock.call_args[0][0], "tokenizer")
            self.assertEqual(fetch_tokenizer_from_hub_mock.call_args[0][1], "tokenizer")
            self.assertEqual(len(fetch_tokenizer_from_hub_mock.call_args[0]), 2)
            is_macos_with_m_chip_mock.assert_called_once()

    @mock_mlx
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
            self.assertEqual(result.exit_code, 0)
            load_mock.assert_called_once()
            load_and_train_mock.assert_called_once()
            convert_between_mlx_and_pytorch_mock.assert_not_called()
            make_data_mock.assert_called_once()
            fetch_tokenizer_from_hub_mock.assert_not_called()
            is_macos_with_m_chip_mock.assert_called_once()

    @patch("instructlab.lab.utils.is_macos_with_m_chip", return_value=False)
    @patch.object(linux_train, "linux_train")
    @patch("instructlab.llamacpp.llamacpp_convert_to_gguf.convert_llama_to_gguf")
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
            result = runner.invoke(lab.train, ["--input-dir", INPUT_DIR])
            self.assertEqual(result.exit_code, 0)
            convert_llama_to_gguf_mock.assert_called_once()
            self.assertEqual(
                convert_llama_to_gguf_mock.call_args[1]["model"],
                "./training_results/final",
            )
            self.assertEqual(convert_llama_to_gguf_mock.call_args[1]["pad_vocab"], True)
            self.assertEqual(len(convert_llama_to_gguf_mock.call_args[1]), 2)
            linux_train_mock.assert_called_once()
            print(linux_train_mock.call_args[1])
            self.assertEqual(
                linux_train_mock.call_args[1]["train_file"],
                "test_generated/train_1.jsonl",
            )
            self.assertEqual(
                linux_train_mock.call_args[1]["test_file"],
                "test_generated/test_1.jsonl",
            )
            self.assertEqual(linux_train_mock.call_args[1]["num_epochs"], 1)
            self.assertIsNotNone(linux_train_mock.call_args[1]["device"])
            self.assertFalse(linux_train_mock.call_args[1]["four_bit_quant"])
            self.assertEqual(len(linux_train_mock.call_args[1]), 5)
            is_macos_with_m_chip_mock.assert_called_once()
            self.assertFalse(os.path.isfile(LINUX_GGUF_FILE))

    @mock_mlx
    @patch("instructlab.lab.utils.is_macos_with_m_chip", return_value=False)
    @patch("instructlab.train.linux_train.linux_train")
    @patch("instructlab.llamacpp.llamacpp_convert_to_gguf.convert_llama_to_gguf")
    def test_num_epochs(
        self, convert_llama_to_gguf_mock, linux_train_mock, is_macos_with_m_chip_mock
    ):
        runner = CliRunner()
        with runner.isolated_filesystem():
            setup_input_dir()
            setup_linux_dir()
            result = runner.invoke(
                lab.train, ["--input-dir", INPUT_DIR, "--num-epochs", "2"]
            )
            self.assertEqual(result.exit_code, 0)
            convert_llama_to_gguf_mock.assert_called_once()
            linux_train_mock.assert_called_once()
            self.assertEqual(linux_train_mock.call_args[1]["num_epochs"], 2)
            is_macos_with_m_chip_mock.assert_called_once()
            self.assertFalse(os.path.isfile(LINUX_GGUF_FILE))

            # Test with invalid num_epochs
            result = runner.invoke(
                lab.train, ["--input-dir", INPUT_DIR, "--num-epochs", "two"]
            )
            self.assertIsNotNone(result.exception)
            self.assertEqual(result.exit_code, 2)
            self.assertIn("'two' is not a valid integer", result.output)
