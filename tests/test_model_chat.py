# Standard
from pathlib import Path
from unittest.mock import MagicMock, patch
import contextlib
import logging

# Third Party
from click.testing import CliRunner
from rich.console import Console
from rich.panel import Panel
import pytest

# First Party
from instructlab import lab
from instructlab.configuration import model_info
from instructlab.feature_gates import FeatureGating, FeatureScopes, GatedFeatures
from instructlab.model.chat import ChatException, ConsoleChatBot
from instructlab.utils import get_sysprompt
from tests.test_feature_gates import dev_preview

# Local
from . import common

logger = logging.getLogger(__name__)


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


def test_rag_chat_errors_with_useful_message_when_not_enabled():
    runner = CliRunner()
    env = runner.make_env({"ILAB_FEATURE_SCOPE": "Default"})
    result = runner.invoke(
        lab.ilab, ["--config=DEFAULT", "model", "chat", "--rag"], env=env
    )

    assert not FeatureGating.feature_available(GatedFeatures.RAG)

    # check that the error message contains the environment variable name and the feature
    # scope level; a (heuristic) check on the message being both up-to-date and useful
    assert FeatureGating.env_var_name in result.output
    assert FeatureScopes.DevPreviewNoUpgrade.value in result.output


@dev_preview
def test_retriever_is_called_when_present():
    retriever = MagicMock()
    chatbot = ConsoleChatBot(
        model="/var/model/file", client=None, retriever=retriever, loaded={}
    )
    assert chatbot.retriever == retriever
    user_query = "test"
    with pytest.raises(ChatException) as exc_info:
        chatbot.start_prompt(content=user_query, logger=logger)
        logger.info(exc_info)
        retriever.augmented_context.assert_called_with(user_query=user_query)


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

    console = Console(force_terminal=False, width=80)
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


@patch("instructlab.cli.model.chat.chat_model")
@patch("instructlab.model.model.storage_dirs_exist", return_value=True)
def test_chat_with_model_id(mock_chat_model, _, cli_runner: CliRunner, tmp_path: Path):
    fname = common.setup_test_models_config(
        models_list=[
            model_info(
                id="test_model",
                path="teacher/model/path",
                family="llama",
                system_prompt="system prompt",
            )
        ],
        dest=tmp_path,
    )

    # Invoke the `chat` command with the model ID
    result = cli_runner.invoke(
        lab.ilab,
        [
            f"--config={tmp_path}/{fname}",
            "model",
            "chat",
            "--model-id=test_model",
        ],
    )

    # Assert that the command executed successfully
    assert result.exit_code == 0, f"Command failed with output: {result.output}"
    mock_chat_model.assert_called_once()


@pytest.mark.parametrize(
    "system_prompt,expected_prompt",
    [
        ("I am a AI assistant.", "I am a AI assistant."),
        (
            None,
            "I am an advanced AI language model designed to assist you with a wide range of tasks and provide helpful, clear, and accurate responses. My primary role is to serve as a chat assistant, engaging in natural, conversational dialogue, answering questions, generating ideas, and offering support across various topics.",
        ),
    ],
)
def test_system_prompt_option(system_prompt, expected_prompt):
    loaded = {}

    chatbot = ConsoleChatBot(
        model="/var/model/file",
        client=None,
        loaded=loaded,
    )

    if system_prompt:
        chatbot.info["messages"] = [{"role": "system", "content": system_prompt}]
    else:
        chatbot.info["messages"] = [
            {"role": "system", "content": get_sysprompt("default")}
        ]

    assert chatbot.info["messages"][0]["content"] == expected_prompt
