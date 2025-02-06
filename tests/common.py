# Standard
from unittest import mock
import pathlib

# Third Party
import yaml

# First Party
from instructlab import configuration, lab

_CFG_FILE_NAME = "test-serve-config.yaml"


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
