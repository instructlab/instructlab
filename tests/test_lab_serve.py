# Standard
from unittest import mock
from unittest.mock import MagicMock, patch
import os
import pathlib

# Third Party
from click.testing import CliRunner
import click
import pytest
import yaml

# First Party
from instructlab import configuration, lab
from instructlab.model.backends.backends import get_max_stable_vram_wait
from instructlab.model.serve import warn_for_unsuported_backend_param

CFG_FILE_NAME = "test-serve-config.yaml"


def vllm_setup_test(mock_popen, mock_determine, runner, args):
    mock_process = mock.MagicMock()
    mock_popen.return_value = mock_process

    mock_process.communicate.return_value = ("out", "err")
    mock_process.returncode = 0
    mock_determine.return_value = ("vllm", "testing")

    result = runner.invoke(lab.ilab, args)
    if result.exit_code != 0:
        print(result.output)

    assert len(mock_popen.call_args_list) == 1
    return mock_popen.call_args_list[0][1]["args"]


def assert_vllm_args(args):
    assert "python" in args[0]
    assert args[1] == "-m"
    assert args[2] == "vllm.entrypoints.openai.api_server"


def assert_template(args, expect_chat, path_chat, chat_value):
    hit_chat = False
    template_exists = False
    template = ""
    for s in args:
        if hit_chat:
            template = s
            template_exists = pathlib.Path(s).exists()
            break  # break as soon as we find the chat template otherwise we will get the next argument and template_exists will fail
        if s == "--chat-template":
            hit_chat = True

    assert hit_chat == expect_chat
    if path_chat:
        assert template_exists
        assert template == chat_value


def assert_tps(args, tps):
    assert args[-2] == "--tensor-parallel-size"
    assert args[-1] == tps


def setup_gpus_config(gpus=None, tps=None):
    cfg = configuration.get_default_config()
    if gpus:
        cfg.serve.vllm.gpus = gpus
    if tps:
        cfg.serve.vllm.vllm_args.extend(["--tensor-parallel-size", str(tps)])
    with pathlib.Path(CFG_FILE_NAME).open("w", encoding="utf-8") as f:
        yaml.dump(cfg.model_dump(), f)


@patch("time.sleep", side_effect=Exception("Intended Abort"))
@patch("instructlab.model.backends.backends.determine_backend")
@patch("subprocess.Popen")
def test_chat_auto(mock_popen, mock_determine, _, cli_runner: CliRunner):
    args = vllm_setup_test(
        mock_popen,
        mock_determine,
        cli_runner,
        ["--config=DEFAULT", "model", "serve", "--model-path=foo"],
    )
    assert_vllm_args(args)
    assert_template(args=args, expect_chat=True, path_chat=False, chat_value="")


@patch("time.sleep", side_effect=Exception("Intended Abort"))
@patch("instructlab.model.backends.backends.determine_backend")
@patch("subprocess.Popen")
def test_chat_custom(
    mock_popen, mock_determine, _, cli_runner: CliRunner, tmp_path: pathlib.Path
):
    file = tmp_path / "chat_template.jinja2"
    file.touch()
    args = vllm_setup_test(
        mock_popen,
        mock_determine,
        cli_runner,
        [
            "--config=DEFAULT",
            "model",
            "serve",
            "--model-path=foo",
            "--chat-template={}".format(str(file)),
        ],
    )
    assert_vllm_args(args)
    assert_template(args=args, expect_chat=True, path_chat=True, chat_value=str(file))


@patch("time.sleep", side_effect=Exception("Intended Abort"))
@patch("instructlab.model.backends.backends.determine_backend")
@patch("subprocess.Popen")
def test_chat_manual(mock_popen, mock_determine, _, cli_runner: CliRunner):
    args = vllm_setup_test(
        mock_popen,
        mock_determine,
        cli_runner,
        [
            "--config=DEFAULT",
            "model",
            "serve",
            "--model-path=foo",
            "--chat-template=tokenizer",
        ],
    )
    assert_vllm_args(args)
    assert_template(args=args, expect_chat=False, path_chat=False, chat_value="")


@patch("time.sleep", side_effect=Exception("Intended Abort"))
@patch("instructlab.model.backends.backends.determine_backend")
@patch("subprocess.Popen")
def test_gpus(mock_popen, mock_determine, _, cli_runner: CliRunner):
    args = vllm_setup_test(
        mock_popen,
        mock_determine,
        cli_runner,
        [
            "--config=DEFAULT",
            "model",
            "serve",
            "--model-path=foo",
            "--gpus",
            "8",
        ],
    )
    assert_tps(args, "8")


@patch("time.sleep", side_effect=Exception("Intended Abort"))
@patch("instructlab.model.backends.backends.determine_backend")
@patch("subprocess.Popen")
def test_gpus_default(mock_popen, mock_determine, _, cli_runner: CliRunner):
    args = vllm_setup_test(
        mock_popen,
        mock_determine,
        cli_runner,
        [
            "--config=DEFAULT",
            "model",
            "serve",
            "--model-path=foo",
        ],
    )
    assert "--tensor-parallel-size" not in args


@patch("time.sleep", side_effect=Exception("Intended Abort"))
@patch("instructlab.model.backends.backends.determine_backend")
@patch("subprocess.Popen")
def test_ctx_tps_with_extra_params(
    mock_popen, mock_determine, _, cli_runner: CliRunner
):
    args = vllm_setup_test(
        mock_popen,
        mock_determine,
        cli_runner,
        [
            "--config=DEFAULT",
            "model",
            "serve",
            "--model-path=foo",
            "--",
            "--served-model-name",
            "mymodel",
            "--tensor-parallel-size",
            "8",
        ],
    )
    assert args[-4] == "--served-model-name"
    assert args[-3] == "mymodel"
    assert_tps(args, "8")


@patch("time.sleep", side_effect=Exception("Intended Abort"))
@patch("instructlab.model.backends.backends.determine_backend")
@patch("subprocess.Popen")
def test_ctx_tps_with_gpus(mock_popen, mock_determine, _, cli_runner: CliRunner):
    args = vllm_setup_test(
        mock_popen,
        mock_determine,
        cli_runner,
        [
            "--config=DEFAULT",
            "model",
            "serve",
            "--model-path=foo",
            "--gpus",
            "8",
            "--",
            "--served-model-name",
            "mymodel",
            "--tensor-parallel-size",
            "4",
        ],
    )
    assert args[-4] == "--served-model-name"
    assert args[-3] == "mymodel"
    assert_tps(args, "4")


@patch("time.sleep", side_effect=Exception("Intended Abort"))
@patch("instructlab.model.backends.backends.determine_backend")
@patch("subprocess.Popen")
def test_gpus_config(mock_popen, mock_determine, _, cli_runner: CliRunner):
    setup_gpus_config(gpus=8)
    args = vllm_setup_test(
        mock_popen,
        mock_determine,
        cli_runner,
        [
            f"--config={CFG_FILE_NAME}",
            "model",
            "serve",
            "--model-path=foo",
        ],
    )
    assert_tps(args, "8")


@patch("time.sleep", side_effect=Exception("Intended Abort"))
@patch("instructlab.model.backends.backends.determine_backend")
@patch("subprocess.Popen")
def test_gpus_with_gpus_config(mock_popen, mock_determine, _, cli_runner: CliRunner):
    setup_gpus_config(gpus=8)
    args = vllm_setup_test(
        mock_popen,
        mock_determine,
        cli_runner,
        [
            f"--config={CFG_FILE_NAME}",
            "model",
            "serve",
            "--model-path=foo",
            "--gpus",
            "4",
        ],
    )
    assert_tps(args, "4")


@patch("time.sleep", side_effect=Exception("Intended Abort"))
@patch("instructlab.model.backends.backends.determine_backend")
@patch("subprocess.Popen")
def test_ctx_tps_with_gpus_config(mock_popen, mock_determine, _, cli_runner: CliRunner):
    setup_gpus_config(gpus=8)
    args = vllm_setup_test(
        mock_popen,
        mock_determine,
        cli_runner,
        [
            f"--config={CFG_FILE_NAME}",
            "model",
            "serve",
            "--model-path=foo",
            "--",
            "--tensor-parallel-size",
            "4",
        ],
    )
    assert_tps(args, "4")


@patch("time.sleep", side_effect=Exception("Intended Abort"))
@patch("instructlab.model.backends.backends.determine_backend")
@patch("subprocess.Popen")
def test_gpus_with_ctx_tps_with_gpus_config(
    mock_popen, mock_determine, _, cli_runner: CliRunner
):
    setup_gpus_config(gpus=8)
    args = vllm_setup_test(
        mock_popen,
        mock_determine,
        cli_runner,
        [
            f"--config={CFG_FILE_NAME}",
            "model",
            "serve",
            "--model-path=foo",
            "--gpus",
            "2",
            "--",
            "--tensor-parallel-size",
            "4",
        ],
    )
    assert_tps(args, "4")


@patch("time.sleep", side_effect=Exception("Intended Abort"))
@patch("instructlab.model.backends.backends.determine_backend")
@patch("subprocess.Popen")
def test_vllm_args_config(mock_popen, mock_determine, _, cli_runner: CliRunner):
    setup_gpus_config(tps=8)
    args = vllm_setup_test(
        mock_popen,
        mock_determine,
        cli_runner,
        [
            f"--config={CFG_FILE_NAME}",
            "model",
            "serve",
            "--model-path=foo",
        ],
    )
    assert_tps(args, "8")


@patch("time.sleep", side_effect=Exception("Intended Abort"))
@patch("instructlab.model.backends.backends.determine_backend")
@patch("subprocess.Popen")
def test_vllm_args_config_with_gpus_config(
    mock_popen, mock_determine, _, cli_runner: CliRunner
):
    setup_gpus_config(gpus=4, tps=8)
    args = vllm_setup_test(
        mock_popen,
        mock_determine,
        cli_runner,
        [
            f"--config={CFG_FILE_NAME}",
            "model",
            "serve",
            "--model-path=foo",
        ],
    )
    assert_tps(args, "4")


@patch("time.sleep", side_effect=Exception("Intended Abort"))
@patch("instructlab.model.backends.backends.determine_backend")
@patch("subprocess.Popen")
def test_vllm_args_config_with_gpus(
    mock_popen, mock_determine, _, cli_runner: CliRunner
):
    setup_gpus_config(tps=8)
    args = vllm_setup_test(
        mock_popen,
        mock_determine,
        cli_runner,
        [
            f"--config={CFG_FILE_NAME}",
            "model",
            "serve",
            "--model-path=foo",
            "--gpus",
            "4",
        ],
    )
    assert_tps(args, "4")


@patch("time.sleep", side_effect=Exception("Intended Abort"))
@patch("instructlab.model.backends.backends.determine_backend")
@patch("subprocess.Popen")
def test_vllm_args_config_with_ctx_tps(
    mock_popen, mock_determine, _, cli_runner: CliRunner
):
    setup_gpus_config(tps=8)
    args = vllm_setup_test(
        mock_popen,
        mock_determine,
        cli_runner,
        [
            f"--config={CFG_FILE_NAME}",
            "model",
            "serve",
            "--model-path=foo",
            "--",
            "--tensor-parallel-size",
            "4",
        ],
    )
    assert_tps(args, "4")


def test_max_stable_vram_wait():
    with mock.patch.dict(os.environ, {"ILAB_MAX_STABLE_VRAM_WAIT": "0"}):
        wait = get_max_stable_vram_wait(10)
        assert wait == 0
    with mock.patch.dict(os.environ, clear=True):
        wait = get_max_stable_vram_wait(10)
        assert wait == 10


@pytest.mark.parametrize(
    "param,expected_call_count",
    [
        ("gpu_layers", 1),
        ("num_threads", 1),
        ("max_ctx_size", 1),
        (
            "supported_param",
            0,
        ),  # Example of a parameter that should not trigger a warning
    ],
)
def test_warn_for_unsuported_backend_param(param, expected_call_count):
    with patch("instructlab.model.serve.logger.warning") as mock_warning:
        # Create a mock click.Context object
        ctx = MagicMock()

        # Set the get_parameter_source to return COMMANDLINE for the tested param
        ctx.get_parameter_source.side_effect = (
            lambda x: click.core.ParameterSource.COMMANDLINE if x == param else None
        )

        # Call the function to test
        warn_for_unsuported_backend_param(ctx)

        # Assert that logger.warning was called the expected number of times
        assert mock_warning.call_count == expected_call_count
        if expected_call_count > 0:
            mock_warning.assert_called_with(
                f"Option '--{param.replace('_','-')}' not supported by the backend."
            )
