# SPDX-License-Identifier: Apache-2.0

# Standard
from unittest.mock import Mock, patch

# Third Party
import git
import pytest
import yaml

# First Party
from instructlab import utils
from instructlab.utils import HTMLTagStripper


class TestUtils:
    """Test collection in instructlab.utils."""

    @patch(
        "instructlab.utils.git_clone_checkout",
        return_value=Mock(spec=git.Repo, working_dir="tests/testdata/temp_repo"),
    )
    def test_get_document(self, git_clone_checkout):
        with open(
            "tests/testdata/knowledge_valid.yaml", "r", encoding="utf-8"
        ) as qnafile:
            documents = utils.get_documents(
                source=yaml.safe_load(qnafile).get("document"),
                skip_checkout=True,
            )
            git_clone_checkout.assert_called_once()
            assert len(documents) == 2


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


def test_html_tag_stripper():
    tag_stripper = HTMLTagStripper()
    assert tag_stripper.strip_html_tags("<table><tr><td>abc") == "abc"
