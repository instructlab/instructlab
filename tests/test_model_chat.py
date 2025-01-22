# Standard
from unittest.mock import MagicMock
import contextlib
import logging
import re

# Third Party
from click.testing import CliRunner
from rich.console import Console
import pytest

# First Party
from instructlab import lab
from instructlab.feature_gates import FeatureGating, FeatureScopes, GatedFeatures
from instructlab.model.chat import ChatException, ConsoleChatBot
from tests.test_feature_gates import dev_preview

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
