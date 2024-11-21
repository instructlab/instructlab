# Standard
from unittest.mock import MagicMock, patch
import contextlib
import os
import re
import shutil

# Third Party
from rich.console import Console
import pytest

# First Party
from instructlab.defaults import DEFAULTS
from instructlab.model.chat import ConsoleChatBot


@pytest.mark.parametrize(
    "model_path,expected_name",
    [
        ("var/model/file", "file"),
        ("var/model/directory/", "directory"),
        ("var/model/directory/////", "directory"),
    ],
)
@patch(
    "instructlab.model.chat.get_model_metadata",
    return_value=[{"version": "unknown"}, True],
)
def test_model_name(get_model_metadata, model_path, expected_name):
    model_path = f"{DEFAULTS.MODELS_DIR}/{model_path}"
    os.makedirs(model_path, exist_ok=True)
    chatbot = ConsoleChatBot(model=model_path, client=None, loaded={})
    assert chatbot.model_name == expected_name
    get_model_metadata.assert_called_once()
    shutil.rmtree(model_path)


def handle_output(output):
    return re.sub(r"\s+", " ", output).strip()


def test_list_contexts_output():
    chatbot = ConsoleChatBot(model="/var/model/file", client=None, loaded={})

    def mock_sys_print(output):
        mock_sys_print.output = output

    chatbot._sys_print = mock_sys_print

    mock_prompt_session = MagicMock()
    mock_prompt_session.prompt.return_value = "/lc"
    chatbot.input = mock_prompt_session

    with contextlib.suppress(KeyboardInterrupt):
        chatbot.start_prompt(logger=None)

    console = Console(force_terminal=False)
    with console.capture() as capture:
        console.print(mock_sys_print.output)

    rendered_output = capture.get().strip()

    expected_output = (
        "Available contexts:\n\n"
        "default: I am an advanced AI language model designed to assist you with a wide range of tasks and provide helpful, clear, and accurate responses. My primary role is to serve as a chat assistant, engaging in natural, conversational dialogue, answering questions, generating ideas, and offering support across various topics.\n\n"
        "cli_helper: You are an expert for command line interface and know all common "
        "commands. Answer the command to execute as it without any explanation."
    )

    assert handle_output(rendered_output) == handle_output(expected_output)
