# Standard
from pathlib import Path
from unittest import mock
import json
import pathlib
import struct

# Third Party
import yaml

# First Party
from instructlab import configuration, lab

_CFG_FILE_NAME = "test-serve-config.yaml"


def setup_test_models_config(
    models_list, dest="", global_student_id=None, global_teacher_id=None
):
    _cfg_name = "test-models-config.yaml"
    cfg = configuration.get_default_config()

    cfg.models = models_list

    if global_student_id:
        cfg.general.student_model_id = global_student_id
    if global_teacher_id:
        cfg.general.teacher_model_id = global_teacher_id

    with pathlib.Path(f"{dest}/{_cfg_name}").open("w", encoding="utf-8") as f:
        yaml.dump(cfg.model_dump(), f)
    return _cfg_name


def setup_gpus_config(section_path="serve", gpus=None, tps=None, vllm_args=lambda: []):
    """Returns the name of the config file with the requested vllm config."""
    cfg = configuration.get_default_config()

    section = None
    for subpath in section_path.split("."):
        section = getattr(section or cfg, subpath)

    section.vllm.vllm_args = vllm_args()
    if gpus:
        section.vllm.gpus = gpus
    if tps:
        section.vllm.vllm_args.extend(["--tensor-parallel-size", str(tps)])

    # TODO: generate the name at random?
    with pathlib.Path(_CFG_FILE_NAME).open("w", encoding="utf-8") as f:
        yaml.dump(cfg.model_dump(), f)

    return _CFG_FILE_NAME


@mock.patch("torch.cuda.device_count", return_value=10)
@mock.patch(
    "instructlab.model.serve_backend.get_tensor_parallel_size",
    return_value=0,
)
@mock.patch(
    "instructlab.model.backends.backends.check_model_path_exists", return_value=None
)
@mock.patch(
    "instructlab.model.backends.backends.determine_backend",
    return_value=("vllm", "testing"),
)
@mock.patch("subprocess.Popen")
def vllm_setup_test(runner, args, mock_popen, *_mock_args):
    mock_process = mock.MagicMock()
    mock_popen.return_value = mock_process

    mock_process.communicate.return_value = ("out", "err")
    mock_process.returncode = 0

    result = runner.invoke(lab.ilab, args)

    if result.exit_code != 0:
        print(result.output)

    vllm_calls = [
        call
        for call in mock_popen.call_args_list
        if "vllm.entrypoints.openai.api_server" in call.kwargs.get("args", [])
    ]
    assert len(vllm_calls) == 1 or (
        "Retrying (1/1)" in result.output and len(vllm_calls) == 2
    )

    return vllm_calls[0][1]["args"]


def assert_tps(args, tps):
    assert args[-2] == "--tensor-parallel-size"
    assert args[-1] == tps


def create_safetensors_model_directory(
    directory_path: Path, model_dir_name="test_namespace/testlab_model"
):
    """Simulate a safetensors model directory"""
    full_directory_path = directory_path / model_dir_name
    full_directory_path.mkdir(parents=True, exist_ok=True)

    json_data = {"key": "value"}
    required_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]

    for file_name in required_files:
        with open(full_directory_path / file_name, "w", encoding="utf-8") as f:
            json.dump(json_data, f)

    safetensors_file = full_directory_path / "test-model.safetensors"
    # Third Party
    from safetensors.torch import save_file
    import torch

    tensors = {
        "tensor1": torch.randn(3, 3),
        "tensor2": torch.randn(5, 5),
    }
    save_file(tensors, safetensors_file)


def create_gguf_file(file_path: Path, gguf_file_name="test-model.gguf"):
    """Simulate a GGUF file"""
    GGUF_MAGIC = 0x46554747

    full_file_path = file_path / gguf_file_name
    full_file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(full_file_path, "wb") as f:
        f.write(struct.pack("<I", GGUF_MAGIC))
