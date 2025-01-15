# Third Party
import pytest

# First Party
from instructlab.rag.haystack import component_factory as f


@pytest.fixture(name="document_store")
def fixture_document_store():
    return f.create_document_store("", "", True)


def test_converter():
    converter = f.create_converter()
    assert converter is not None
    assert type(converter).__name__ == "TextFileToDocument"


def test_document_embedder():
    embedder = f.create_document_embedder(embedding_model_path="foo")
    assert embedder is not None
    assert type(embedder).__name__ == "SentenceTransformersDocumentEmbedder"


def test_text_embedder():
    embedder = f.create_text_embedder(embedding_model_path="foo")
    assert embedder is not None
    assert type(embedder).__name__ == "SentenceTransformersTextEmbedder"


def test_retriever(document_store):
    retriever = f.create_retriever(top_k=10, document_store=document_store)
    assert retriever is not None
    assert type(retriever).__name__ == "InMemoryEmbeddingRetriever"


def test_document_store(document_store):
    assert document_store is not None
    assert type(document_store).__name__ == "InMemoryDocumentStore"


def test_document_writer(document_store):  # pylint: disable=unused-argument
    writer = f.create_document_writer("", "")
    assert writer is not None
    assert type(writer).__name__ == "DocumentWriter"


def test_document_splitter():
    with pytest.raises(OSError) as _:
        f.create_splitter("foo")
