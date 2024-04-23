# SPDX-License-Identifier: Apache-2.0

# Standard
from unittest.mock import Mock, patch
import logging
import unittest

# Third Party
import git
import yaml

# First Party
from instructlab import utils

# Local
from .testdata import testdata


class TestUtils(unittest.TestCase):
    """Test collection in instructlab.utils."""

    def test_chunk_docs_wc_exeeds_ctx_window(self):
        with self.assertRaises(ValueError) as exc:
            utils.chunk_document(
                documents=testdata.documents,
                chunk_word_count=1000,
                server_ctx_size=1034,
            )
        self.assertIn(
            "Given word count (1000) per doc will exceed the server context window size (1034)",
            f"{exc.exception}",
        )

    def test_chunk_docs_chunk_overlap_error(self):
        with self.assertRaises(ValueError) as exc:
            utils.chunk_document(
                documents=testdata.documents,
                chunk_word_count=5,
                server_ctx_size=1034,
            )
        self.assertIn(
            "Got a larger chunk overlap (100) than chunk size (24), should be smaller",
            f"{exc.exception}",
        )

    def test_chunk_docs_long_lines(self):
        chunk_words = 50
        chunks = utils.chunk_document(
            documents=testdata.long_line_documents,
            chunk_word_count=chunk_words,
            server_ctx_size=4096,
        )
        # convert words to characters as done in chunk_document
        max_chunk_length = chunk_words * 1.3 * 4
        max_chunk_length += 100  # add in the chunk overlap
        max_chunk_length += 50  # and a bit extra for some really long words
        for chunk in chunks:
            assert len(chunk) <= max_chunk_length

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
                logger=logging.getLogger("_test_"),
            )
            git_clone_checkout.assert_called_once()
            self.assertEqual(len(documents), 2)
