# Standard
from unittest import mock
from unittest.mock import MagicMock, patch
import os
import pathlib

# Third Party
from click.testing import CliRunner
import click
import pytest

# First Party
from instructlab.cli.model.serve import warn_for_unsupported_backend_param
from instructlab.model.backends.vllm import get_max_stable_vram_wait

# Local
from . import common


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


def test_chat_auto(cli_runner: CliRunner):
    args = common.vllm_setup_test(
        cli_runner,
        ["--config=DEFAULT", "model", "serve", "--model-path=foo"],
    )
    assert_vllm_args(args)
    assert_template(args=args, expect_chat=True, path_chat=False, chat_value="")


def test_chat_custom(cli_runner: CliRunner, tmp_path: pathlib.Path):
    file = tmp_path / "chat_template.jinja2"
    file.touch()
    args = common.vllm_setup_test(
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


def test_chat_manual(cli_runner: CliRunner):
    args = common.vllm_setup_test(
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


def test_gpus(cli_runner: CliRunner):
    args = common.vllm_setup_test(
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
    common.assert_tps(args, "8")


def test_gpus_default(cli_runner: CliRunner):
    args = common.vllm_setup_test(
        cli_runner,
        [
            "--config=DEFAULT",
            "model",
            "serve",
            "--model-path=foo",
        ],
    )
    assert "--tensor-parallel-size" not in args


def test_ctx_tps_with_extra_params(cli_runner: CliRunner):
    args = common.vllm_setup_test(
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
    common.assert_tps(args, "8")


def test_ctx_tps_with_gpus(cli_runner: CliRunner):
    args = common.vllm_setup_test(
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
    common.assert_tps(args, "4")


def test_gpus_config(cli_runner: CliRunner):
    fname = common.setup_gpus_config(gpus=8)
    args = common.vllm_setup_test(
        cli_runner,
        [
            f"--config={fname}",
            "model",
            "serve",
            "--model-path=foo",
        ],
    )
    common.assert_tps(args, "8")


def test_gpus_with_gpus_config(cli_runner: CliRunner):
    fname = common.setup_gpus_config(gpus=8)
    args = common.vllm_setup_test(
        cli_runner,
        [
            f"--config={fname}",
            "model",
            "serve",
            "--model-path=foo",
            "--gpus",
            "4",
        ],
    )
    common.assert_tps(args, "4")


def test_ctx_tps_with_gpus_config(cli_runner: CliRunner):
    fname = common.setup_gpus_config(gpus=8)
    args = common.vllm_setup_test(
        cli_runner,
        [
            f"--config={fname}",
            "model",
            "serve",
            "--model-path=foo",
            "--",
            "--tensor-parallel-size",
            "4",
        ],
    )
    common.assert_tps(args, "4")


def test_gpus_with_ctx_tps_with_gpus_config(cli_runner: CliRunner):
    fname = common.setup_gpus_config(gpus=8)
    args = common.vllm_setup_test(
        cli_runner,
        [
            f"--config={fname}",
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
    common.assert_tps(args, "4")


def test_vllm_args_config(cli_runner: CliRunner):
    fname = common.setup_gpus_config(tps=8)
    args = common.vllm_setup_test(
        cli_runner,
        [
            f"--config={fname}",
            "model",
            "serve",
            "--model-path=foo",
        ],
    )
    common.assert_tps(args, "8")


def test_vllm_args_config_with_gpus_config(cli_runner: CliRunner):
    fname = common.setup_gpus_config(gpus=4, tps=8)
    args = common.vllm_setup_test(
        cli_runner,
        [
            f"--config={fname}",
            "model",
            "serve",
            "--model-path=foo",
        ],
    )
    common.assert_tps(args, "4")


def test_vllm_args_config_with_gpus(cli_runner: CliRunner):
    fname = common.setup_gpus_config(tps=8)
    args = common.vllm_setup_test(
        cli_runner,
        [
            f"--config={fname}",
            "model",
            "serve",
            "--model-path=foo",
            "--gpus",
            "4",
        ],
    )
    common.assert_tps(args, "4")


def test_vllm_args_config_with_ctx_tps(cli_runner: CliRunner):
    fname = common.setup_gpus_config(tps=8)
    args = common.vllm_setup_test(
        cli_runner,
        [
            f"--config={fname}",
            "model",
            "serve",
            "--model-path=foo",
            "--",
            "--tensor-parallel-size",
            "4",
        ],
    )
    common.assert_tps(args, "4")


def test_vllm_args_null(cli_runner: CliRunner):
    fname = common.setup_gpus_config(vllm_args=lambda: None)
    args = common.vllm_setup_test(
        cli_runner,
        [
            f"--config={fname}",
            "model",
            "serve",
            "--model-path=foo",
            "--gpus",
            "4",
        ],
    )
    common.assert_tps(args, "4")


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
def test_warn_for_unsupported_backend_param(param, expected_call_count):
    with patch("instructlab.cli.model.serve.logger.warning") as mock_warning:
        # Create a mock click.Context object
        ctx = MagicMock()

        # Set the get_parameter_source to return COMMANDLINE for the tested param
        ctx.get_parameter_source.side_effect = (
            lambda x: click.core.ParameterSource.COMMANDLINE if x == param else None
        )

        # Call the function to test
        warn_for_unsupported_backend_param(ctx)

        # Assert that logger.warning was called the expected number of times
        assert mock_warning.call_count == expected_call_count
        if expected_call_count > 0:
            mock_warning.assert_called_with(
                f"Option '--{param.replace('_','-')}' not supported by the backend."
            )


def test_serve_host(cli_runner: CliRunner):
    args = common.vllm_setup_test(
        cli_runner,
        [
            "--config=DEFAULT",
            "model",
            "serve",
            "--model-path=foo",
            "--host",
            "2.4.6.8",
        ],
    )
    assert "2.4.6.8" in args
    assert "8000" in args


def test_serve_port(cli_runner: CliRunner):
    args = common.vllm_setup_test(
        cli_runner,
        [
            "--config=DEFAULT",
            "model",
            "serve",
            "--model-path=foo",
            "--port",
            "1234",
        ],
    )
    assert "127.0.0.1" in args
    assert "1234" in args


def test_serve_host_port(cli_runner: CliRunner):
    args = common.vllm_setup_test(
        cli_runner,
        [
            "--config=DEFAULT",
            "model",
            "serve",
            "--model-path=foo",
            "--host",
            "192.168.1.1",
            "--port",
            "1234",
        ],
    )
    assert "192.168.1.1" in args
    assert "1234" in args
