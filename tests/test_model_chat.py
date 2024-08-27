# Third Party
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
