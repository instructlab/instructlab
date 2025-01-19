# Standard
from pathlib import Path
from unittest.mock import patch
import platform

# Third Party
from click.testing import CliRunner
import pytest

# First Party
from instructlab import lab


@pytest.mark.skipif(
    platform.system() != "Darwin", reason="The test is only run on MacOS"
)
@patch(
    "instructlab.llamacpp.llamacpp_convert_to_gguf.convert_llama_to_gguf",
    return_value="outfile",
)
@patch("instructlab.train.lora_mlx.convert.convert_between_mlx_and_pytorch")
@patch("instructlab.train.lora_mlx.fuse.fine_tune")
def test_model_convert_model_dir_slash(
    fine_tune_mock,
    convert_between_mlx_and_pytorch_mock,
    convert_llama_to_gguf_mock,
    cli_runner: CliRunner,
    tmp_path: Path,
):
    model_dir_with_slash = str(tmp_path) + "/instructlab-granite-7b-lab-mlx-q/"
    adapter_file = model_dir_with_slash + "adapters-100.npz"
    # model_dir_fused dir is necessary to create during the test
    # it also fail to create dir if with slash for model_dir_fused
    model_dir_fused_str = model_dir_with_slash.rstrip("/")
    model_dir_fused = Path(str(model_dir_fused_str) + "-fused")
    model_dir_fused.mkdir()

    result = cli_runner.invoke(
        lab.ilab,
        [
            "--config=DEFAULT",
            "model",
            "convert",
            "--adapter-file",
            adapter_file,
            "--model-dir",
            model_dir_with_slash,
        ],
    )

    assert fine_tune_mock.call_count == 1
    assert convert_llama_to_gguf_mock.call_count == 1
    assert convert_between_mlx_and_pytorch_mock.call_count == 1
    # model_name without -mlx-q e.g. from instructlab-granite-7b-lab-mlx-q to instructlab-granite-7b-lab
    expected_model_fused_pt = "instructlab-granite-7b-lab-trained"
    assert f"{expected_model_fused_pt}" in result.output


@pytest.mark.skipif(
    platform.system() != "Darwin", reason="The test is only run on MacOS"
)
@patch(
    "instructlab.llamacpp.llamacpp_convert_to_gguf.convert_llama_to_gguf",
    return_value="outfile",
)
@patch("instructlab.train.lora_mlx.convert.convert_between_mlx_and_pytorch")
@patch("instructlab.train.lora_mlx.fuse.fine_tune")
def test_model_convert_model_dir_without_slash(
    fine_tune_mock,
    convert_between_mlx_and_pytorch_mock,
    convert_llama_to_gguf_mock,
    cli_runner: CliRunner,
    tmp_path: Path,
):
    model_dir_without_slash = str(tmp_path) + "/instructlab-granite-7b-lab-mlx-q"
    adapter_file = model_dir_without_slash + "adapters-100.npz"
    # model_dir_fused dir is necessary to create during the test
    model_dir_fused = Path(str(model_dir_without_slash) + "-fused")
    model_dir_fused.mkdir()

    result = cli_runner.invoke(
        lab.ilab,
        [
            "--config=DEFAULT",
            "model",
            "convert",
            "--adapter-file",
            adapter_file,
            "--model-dir",
            model_dir_without_slash,
        ],
    )

    assert fine_tune_mock.call_count == 1
    assert convert_llama_to_gguf_mock.call_count == 1
    assert convert_between_mlx_and_pytorch_mock.call_count == 1
    expected_model_fused_pt = "instructlab-granite-7b-lab-trained"
    assert f"{expected_model_fused_pt}" in result.output
