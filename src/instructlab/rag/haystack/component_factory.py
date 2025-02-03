# Standard

# Third Party
from haystack.components.converters import TextFileToDocument  # type: ignore
from haystack.components.embedders import (  # type: ignore
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.retrievers import InMemoryEmbeddingRetriever  # type: ignore
from haystack.components.writers import DocumentWriter  # type: ignore
from haystack.document_stores.in_memory import InMemoryDocumentStore  # type: ignore

# First Party
from instructlab.rag.haystack.components.document_splitter import (
    DoclingDocumentSplitter,
)


def create_document_writer(
    document_store_uri: str, document_store_collection_name: str
) -> DocumentWriter:
    return DocumentWriter(
        create_document_store(
            document_store_uri=document_store_uri,
            document_store_collection_name=document_store_collection_name,
            drop_old=True,
        )
    )


def create_document_store(
    document_store_uri: str, document_store_collection_name: str, drop_old: bool
):
    if not drop_old:
        # Retrieve use case: load from file
        return InMemoryDocumentStore.load_from_disk(document_store_uri)
    return InMemoryDocumentStore()


def create_retriever(
    top_k: int,
    document_store: InMemoryDocumentStore,
):
    return InMemoryEmbeddingRetriever(
        document_store=document_store,
        top_k=top_k,
    )


def create_document_embedder(embedding_model_path: str):
    return SentenceTransformersDocumentEmbedder(model=embedding_model_path)


def create_text_embedder(embedding_model_path: str):
    return SentenceTransformersTextEmbedder(model=embedding_model_path)


def create_converter():
    return TextFileToDocument()


def create_splitter(embedding_model_path: str):
    return DoclingDocumentSplitter(
        embedding_model_id=embedding_model_path,
        content_format="json",
        max_tokens=150,
    )
