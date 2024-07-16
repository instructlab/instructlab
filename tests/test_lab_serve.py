# Standard
from unittest import mock
from unittest.mock import patch
import pathlib

# Third Party
from click.testing import CliRunner

# First Party
from instructlab import lab


def vllm_setup_test(mock_popen, mock_determine, runner, args):
    mock_process = mock.MagicMock()
    mock_popen.return_value = mock_process

    mock_process.communicate.return_value = ("out", "err")
    mock_process.returncode = 0
    mock_determine.return_value = ("vllm", "testing")

    runner.invoke(lab.ilab, args)

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
