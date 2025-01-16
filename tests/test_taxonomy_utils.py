# Standard
import os
import tempfile
import time

# First Party
from instructlab.rag.taxonomy_utils import lookup_processed_documents_folder


def test_lookup_processed_documents_folder_finds_latest_folder():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create folder structure to navigate into
        os.makedirs(os.path.join(tmpdir, "first/docling-artifacts"))
        time.sleep(1)
        os.makedirs(os.path.join(tmpdir, "second/docling-artifacts"))

        latest_doc_folder = lookup_processed_documents_folder(tmpdir)

        assert latest_doc_folder is not None
        assert latest_doc_folder.parent.name == "second"


def test_lookup_processed_documents_folder_returns_None_when_no_documents():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create folder structure to navigate into
        os.makedirs(os.path.join(tmpdir, "first/docling-artifacts"))
        time.sleep(1)
        os.makedirs(os.path.join(tmpdir, "second"))

        latest_doc_folder = lookup_processed_documents_folder(tmpdir)

        assert latest_doc_folder is None
