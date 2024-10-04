# SPDX-License-Identifier: Apache-2.0

# Standard
from unittest.mock import Mock, patch
import typing

# Third Party
import git
import pytest
import yaml

# First Party
from instructlab import utils
from instructlab.utils import Message, MessageSample


class TestUtils:
    """Test collection in instructlab.utils."""

    @patch(
        "instructlab.utils.git_clone_checkout",
        return_value=Mock(
            spec=git.Repo, working_dir="tests/testdata/temp_taxonomy_repo"
        ),
    )
    def test_validate_documents(self, git_clone_checkout):
        with open(
            "tests/testdata/knowledge_valid.yaml", "r", encoding="utf-8"
        ) as qnafile:
            utils._validate_documents(
                source=yaml.safe_load(qnafile).get("document"),
                skip_checkout=True,
            )
            git_clone_checkout.assert_called_once()

    def test_convert_to_legacy_from_pretraining_messages(
        self,
    ):
        new_dataset: typing.List[MessageSample] = [
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a friendly assistant",
                    },
                    {
                        "role": "pretraining",
                        "content": "<|user|>What is 2+2?<|assistant|>2+2=4",
                    },
                ],
                "group": " test",
                "dataset": "test-dataset",
                "metadata": "test",
            }
        ]
        legacy = utils.ensure_legacy_dataset(new_dataset)
        assert len(legacy) == 1
        assert legacy[0]["system"] == "You are a friendly assistant"
        assert legacy[0]["user"] == "What is 2+2?"
        assert legacy[0]["assistant"] == "2+2=4"

    @pytest.mark.parametrize(
        "content,exception,match",
        [
            ("<|user|>What is 2+2? 2+2=4", ValueError, "<|assistant|>"),
            ("<|assistant|>2+2=4", ValueError, "<|user|>"),
            ("<|user|>what is 2+2?<|assistant|>2+2=4", None, ""),
        ],
    )
    def test_invalid_pretraining_messages(
        self, content: str, exception: Exception | None, match: str
    ):
        new_dataset: typing.List[MessageSample] = [
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a friendly assistant",
                    },
                    {
                        "role": "pretraining",
                        "content": content,
                    },
                ],
                "group": " test",
                "dataset": "test-dataset",
                "metadata": "test",
            }
        ]
        if exception:
            with pytest.raises(ValueError, match=match):
                utils.ensure_legacy_dataset(new_dataset)
        else:
            utils.ensure_legacy_dataset(new_dataset)

    def test_pretraining_messages_without_system(self):
        new_dataset: typing.List[MessageSample] = [
            {
                "messages": [
                    {
                        "role": "pretraining",
                        "content": "<|user|>What is 2+2?<|assistant|>2+2=4",
                    },
                ],
                "group": " test",
                "dataset": "test-dataset",
                "metadata": "test",
            }
        ]
        legacy = utils.ensure_legacy_dataset(new_dataset)
        assert len(legacy) == 1
        assert legacy[0]["system"] == ""

    def test_convert_to_legacy_from_messages(self):
        messages: typing.List[MessageSample] = [
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a friendly assistant",
                    },
                    {"role": "user", "content": "Who is pickle rick?"},
                    {
                        "role": "assistant",
                        "content": "As an AI language model, I have absolutely no idea.",
                    },
                ],
                "group": " test",
                "dataset": "test-dataset",
                "metadata": "test",
            }
        ]
        legacy = utils.ensure_legacy_dataset(messages)
        assert len(legacy) == 1
        sample = legacy[0]
        assert sample["system"] == "You are a friendly assistant"
        assert sample["user"] == "Who is pickle rick?"
        assert (
            sample["assistant"] == "As an AI language model, I have absolutely no idea."
        )

    @pytest.mark.parametrize(
        "system,user,assistant",
        [
            (None, None, None),
            ("You are a friendly assistant trained by ACME corp", None, None),
            (None, "Who is pickle rick?", None),
            (
                "You are a friendly assistant trained by ACME corp",
                "Who is pickle rick?",
                None,
            ),
            (None, None, "As an AI language model, I have absolutely no idea."),
            (
                "You are a friendly assistant trained by ACME corp",
                None,
                "As an AI language model, I have absolutely no idea.",
            ),
            (
                None,
                "Who is pickle rick?",
                "As an AI language model, I have absolutely no idea.",
            ),
        ],
    )
    def test_invalid_datasets(
        self, system: str | None, user: str | None, assistant: str | None
    ):
        messages: typing.List[Message] = []
        if system:
            messages.append({"content": system, "role": "system"})
        if user:
            messages.append({"content": user, "role": "user"})
        if assistant:
            messages.append({"content": assistant, "role": "assistant"})
        dataset: typing.List[MessageSample] = [
            {
                "messages": messages,
                "group": "ACME",
                "dataset": "The Pickle Rick Collection",
                "metadata": "{{ pickle: rick, }}",
            }
        ]
        with pytest.raises(ValueError):
            utils.ensure_legacy_dataset(dataset)


@pytest.mark.parametrize(
    "url,expected_host,expected_port",
    [
        ("127.0.0.1:8080", "127.0.0.1", 8080),
        ("[::1]:8080", "::1", 8080),
        ("host.test:9090", "host.test", 9090),
        ("https://host.test:443/egg/spam", "host.test", 443),
    ],
)
def test_split_hostport(url, expected_host, expected_port):
    host, port = utils.split_hostport(url)
    assert host == expected_host
    assert port == expected_port


@pytest.mark.parametrize(
    "url",
    [
        "127.0.0.1",
        "",
        "::1:8080",
    ],
)
def test_split_hostport_err(url):
    with pytest.raises(ValueError):
        utils.split_hostport(url)
