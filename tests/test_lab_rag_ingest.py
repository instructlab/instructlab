# Standard
from pathlib import Path
from typing import Iterator
from unittest.mock import patch
import os
import shutil

# Third Party
from click.testing import CliRunner
from docling_core.transforms.chunker.hybrid_chunker import BaseChunk, DocChunk, DocMeta
from docling_core.types import DoclingDocument
from docling_core.types.doc import DocItem, DocItemLabel, DocumentOrigin
from haystack import Document  # type: ignore
import pytest

# First Party
from instructlab import lab
from instructlab.defaults import DEFAULTS
from instructlab.feature_gates import FeatureGating, FeatureScopes, GatedFeatures
from instructlab.rag.document_store import DocumentStoreIngestor
from instructlab.rag.haystack.components.document_splitter import (
    DoclingDocumentSplitter,
)
from tests.test_feature_gates import dev_preview


class MockDocumentStoreIngestor(DocumentStoreIngestor):
    def __init__(self, document_store_uri: str):
        self.document_store_uri = document_store_uri

    def ingest_documents(self, input_dir: str) -> tuple[bool, int]:
        with open(self.document_store_uri, "w", encoding="utf-8") as _:
            return (True, 0)


# Parameters: input_dir str and completion status (True if succeeds)
@pytest.fixture(
    name="input_dir",
    params=[("tests/testdata/temp_datasets_documents", True), (None, False)],
)
def fixture_input_dir(request):
    return request.param


@pytest.fixture(name="copy_test_data")
def fixture_copy_test_data(input_dir: str, tmp_path_home, cli_runner: CliRunner):  # pylint: disable=unused-argument
    if input_dir[0] is not None:
        tmp_dir = os.path.join(tmp_path_home, os.listdir(tmp_path_home)[0])
        destination = os.path.join(tmp_dir, input_dir[0])
        os.makedirs(destination, exist_ok=True)
        shutil.copytree(input_dir[0], destination, dirs_exist_ok=True)


def test_rag_ingest_errors_with_useful_message_when_not_enabled():
    runner = CliRunner()
    env = runner.make_env({"ILAB_FEATURE_SCOPE": "Default"})
    result = runner.invoke(lab.ilab, ["--config=DEFAULT", "rag", "ingest"], env=env)

    assert not FeatureGating.feature_available(GatedFeatures.RAG)

    # check that the error message contains the environment variable name and the feature
    # scope level; a (heuristic) check on the message being both up-to-date and useful
    assert FeatureGating.env_var_name in result.output
    assert FeatureScopes.DevPreviewNoUpgrade.value in result.output


@patch(
    "instructlab.rag.document_store_factory.create_document_store_ingestor",
    side_effect=(
        lambda document_store_uri,
        document_store_collection_name,
        embedding_model_path: MockDocumentStoreIngestor(document_store_uri)
    ),
)
@dev_preview
def test_ingestor(
    create_document_store_ingestor_mock,
    input_dir,
    tmp_path_home,
    copy_test_data,
    cli_runner: CliRunner,
):  # pylint: disable=unused-argument
    document_store_config_uri = os.path.join(tmp_path_home, "any.db")
    assert not Path(document_store_config_uri).exists()

    result = cli_runner.invoke(
        lab.ilab,
        [
            "--config=DEFAULT",
            "rag",
            "ingest",
            "--document-store-uri",
            document_store_config_uri,
            "--input-dir",
            input_dir[0],
        ],
    )

    if input_dir[1] is True:
        create_document_store_ingestor_mock.assert_called_once()

        assert Path(document_store_config_uri).exists()
        assert result.exit_code == 0
    else:
        assert result.exit_code == 1


class MockHybridChunker:
    def __init__(
        self,
        tokenizer: str,
        max_tokens: int,
    ):
        pass

    def chunk(self, dl_doc: DoclingDocument) -> Iterator[BaseChunk]:
        meta = DocMeta(
            doc_items=[
                DocItem(
                    label=DocItemLabel.TEXT,
                    self_ref="#/texts/1",
                )
            ],
            headings=["header"],
            captions=["caption"],
            origin=DocumentOrigin(
                mimetype="application/pdf", binary_hash=1234567890, filename="test.pdf"
            ),
        )
        return iter(
            [
                DocChunk(meta=meta, text=dl_doc.export_to_text()),
            ]
        )

    def serialize(self, chunk: BaseChunk) -> str:
        return chunk.text


def test_document_splitter_chunks():
    with patch(
        "instructlab.rag.haystack.components.document_splitter.HybridChunker",
        MockHybridChunker,
    ):
        splitter = DoclingDocumentSplitter(
            embedding_model_id="foo",
            content_format=DEFAULTS.SUPPORTED_CONTENT_FORMATS[0],
            max_tokens=512,
        )
        file_path = "tests/testdata/temp_datasets_documents/docling-artifacts/knowledge-wiki.json"
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        doc = Document(content=text, meta={"file_path": file_path})
        result = splitter.run([doc])
        print(result)

        assert result is not None
        assert "documents" in result
        docs = result["documents"]
        assert isinstance(docs, list)
        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        doc = docs[0]
        assert "schema_name" not in doc.meta
        assert "version" not in doc.meta
        assert "doc_items" not in doc.meta
        assert "headings" in doc.meta
        assert "captions" in doc.meta
        assert "origin" in doc.meta
        origin = doc.meta["origin"]
        assert "mimetype" in origin
        assert "binary_hash" in origin
        assert "filename" in origin
        assert origin["filename"] == "test.pdf"
