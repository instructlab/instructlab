# Standard
from typing import Iterator
from unittest.mock import patch
import glob

# Third Party
from docling_core.transforms.chunker import BaseChunk, HierarchicalChunker
from docling_core.types import DoclingDocument
from haystack.components.converters import TextFileToDocument  # type: ignore
import pytest

# First Party
from instructlab.rag.haystack.components.document_splitter import (
    DoclingDocumentSplitter,
)


class MockChunker:
    def __init__(
        self,
    ) -> None:
        self.chunker = HierarchicalChunker()

    def chunk(self, dl_doc: DoclingDocument) -> Iterator[BaseChunk]:
        return self.chunker.chunk(dl_doc=dl_doc)

    def serialize(self, chunk: BaseChunk) -> str:
        return self.chunker.serialize(chunk=chunk)


@pytest.fixture(name="document_splitter")  # pylint: disable=unused-argument
def fixture_document_splitter():
    with patch(
        "instructlab.rag.haystack.components.document_splitter.HybridChunker",
        side_effect=lambda tokenizer, max_tokens: MockChunker(),
    ):
        return DoclingDocumentSplitter(content_format="json")


def test_document_splitter(document_splitter):
    converter = TextFileToDocument()
    sources = glob.glob(
        "tests/testdata/temp_datasets_documents/docling-artifacts/*.json"
    )
    docs = converter.run(sources=sources)
    assert "documents" in docs

    result = document_splitter.run(documents=docs["documents"])

    assert result is not None
    assert isinstance(result, dict)
    assert "documents" in result
    docs = result["documents"]
    assert len(docs) > 0


def test_wrong_documents_type(document_splitter):
    with pytest.raises(TypeError):
        document_splitter.run(documents=None)


def test_document_splitter_serialization(document_splitter):
    _dict = document_splitter.to_dict()
    assert _dict is not None
    assert isinstance(_dict, dict)

    with patch(
        "instructlab.rag.haystack.components.document_splitter.HybridChunker",
        side_effect=lambda tokenizer, max_tokens: MockChunker(),
    ):
        _splitter = DoclingDocumentSplitter.from_dict(data=_dict)
        assert _splitter is not None
        assert isinstance(_splitter, DoclingDocumentSplitter)
