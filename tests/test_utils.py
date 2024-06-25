# SPDX-License-Identifier: Apache-2.0

# Standard
from unittest.mock import Mock, patch

# Third Party
import git
import yaml

# First Party
from instructlab import utils


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
