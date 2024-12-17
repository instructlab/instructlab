# Standard
from unittest.mock import MagicMock
import contextlib

# Third Party
from rich.console import Console
from rich.panel import Panel
import pytest

# First Party
from instructlab.model.chat import ConsoleChatBot


@pytest.mark.parametrize(
    "model_path,expected_name",
    [
        ("/var/model/file", "file"),
        ("/var/model/directory/", "directory"),
        ("/var/model/directory/////", "directory"),
    ],
)
def test_model_name(model_path, expected_name):
    chatbot = ConsoleChatBot(model=model_path, client=None, loaded={})
    assert chatbot.model_name == expected_name


def test_list_contexts_and_decoration():
    chatbot = ConsoleChatBot(model="/var/model/file", client=None, loaded={})

    def mock_sys_print_output(*args, **kwargs):
        if chatbot.box:
            panel = Panel(*args, **kwargs)
            mock_sys_print_output.output = panel
        else:
            mock_sys_print_output.output = args[0]

    chatbot._sys_print = mock_sys_print_output

    # Test when box=True
    chatbot.box = True
    mock_prompt_session = MagicMock()
    mock_prompt_session.prompt.return_value = "/lc"
    chatbot.input = mock_prompt_session

    with contextlib.suppress(KeyboardInterrupt):
        chatbot.start_prompt(logger=None)

    console = Console(force_terminal=False)
    with console.capture() as capture:
        console.print(mock_sys_print_output.output)

    rendered_output = capture.get().strip()

    expected_output_with_box = (
        "╭──────────────────────────────────────────────────────────────────────────────╮\n"
        "│ Available contexts:                                                          │\n"
        "│                                                                              │\n"
        "│ default: I am an advanced AI language model designed to assist you with a    │\n"
        "│ wide range of tasks and provide helpful, clear, and accurate responses. My   │\n"
        "│ primary role is to serve as a chat assistant, engaging in natural,           │\n"
        "│ conversational dialogue, answering questions, generating ideas, and offering │\n"
        "│ support across various topics.                                               │\n"
        "│                                                                              │\n"
        "│ cli_helper: You are an expert for command line interface and know all common │\n"
        "│ commands. Answer the command to execute as it without any explanation.       │\n"
        "╰──────────────────────────────────────────────────────────────────────────────╯"
    )

    assert rendered_output == expected_output_with_box

    # Test when box=False
    chatbot.box = False
    mock_prompt_session = MagicMock()
    mock_prompt_session.prompt.return_value = "/lc"
    chatbot.input = mock_prompt_session

    with contextlib.suppress(KeyboardInterrupt):
        chatbot.start_prompt(logger=None)

    with console.capture() as capture:
        console.print(mock_sys_print_output.output)

    rendered_output = capture.get().strip()

    expected_output_without_box = (
        "Available contexts:                                                             \n\n"
        "default: I am an advanced AI language model designed to assist you with a wide  \n"
        "range of tasks and provide helpful, clear, and accurate responses. My primary   \n"
        "role is to serve as a chat assistant, engaging in natural, conversational       \n"
        "dialogue, answering questions, generating ideas, and offering support across    \n"
        "various topics.                                                                 \n\n"
        "cli_helper: You are an expert for command line interface and know all common    \n"
        "commands. Answer the command to execute as it without any explanation."
    )

    assert rendered_output == expected_output_without_box
