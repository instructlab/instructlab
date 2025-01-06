# Standard
from unittest.mock import MagicMock, patch
import contextlib
import re

# Third Party
from rich.console import Console
import pytest

# First Party
from instructlab.model.chat import ConsoleChatBot, process_prompts_from_file


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


class MockCCB:
    def _reset_session(self):
        pass


@patch("instructlab.model.chat.process_prompt")
def test_process_prompts_from_file(mock_process_prompt, tmp_path):
    def mock_process(_ccb, prompt, output=None):
        response = f"Response for {prompt}"
        if output:
            output.write(f"Q: {prompt}\nA: {response}\n\n")
        return response

    mock_process_prompt.side_effect = mock_process

    prompt_file = tmp_path / "prompts.txt"
    output_file = tmp_path / "output.txt"

    prompt_file.write_text("Prompt 1\nPrompt 2\n")

    assert prompt_file.stat().st_size > 0

    ccb = MockCCB()
    process_prompts_from_file(ccb, prompt_file, output_file)

    output_content = output_file.read_text()

    output_lines = output_content.splitlines()

    assert "Q: Prompt 1\nA: Response for Prompt 1\n" in "\n".join(output_lines)
    assert "Q: Prompt 2\nA: Response for Prompt 2\n" in "\n".join(output_lines)
