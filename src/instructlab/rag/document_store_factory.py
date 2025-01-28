# Standard
import logging

# First Party
from instructlab.rag.document_store import DocumentStoreIngestor, DocumentStoreRetriever

logger = logging.getLogger(__name__)


def create_document_retriever(
    document_store_uri: str,
    document_store_collection_name: str,
    top_k: int,
    embedding_model_path: str,
) -> DocumentStoreRetriever:
    """
    Creates a `DocumentStoreRetriever` instance using the provided settings.

    Args:
        document_store_uri: URI of the document store service (or file if it's an embedded instance).
        document_store_collection_name: Name of the document store collection from which the embeddings are retrieved.
        top_k: Number of documents to retrieve at each request.
        embedding_model_path: Path of the embedding model used to generate the query embeddings.

    Returns:
        An instance of `DocumentStoreRetriever` according to the provided settings.
    """

    # First Party
    from instructlab.rag.haystack.document_store_factory import (
        create_in_memory_document_retriever,
    )

    if not logger.isEnabledFor(logging.DEBUG):
        # Increase log level for RAG components
        for _logger in [
            logging.getLogger(name) for name in ["haystack", "sentence_transformers"]
        ]:
            _logger.setLevel(logging.WARNING)

    return create_in_memory_document_retriever(
        document_store_uri=document_store_uri,
        document_store_collection_name=document_store_collection_name,
        top_k=top_k,
        embedding_model_path=embedding_model_path,
    )


def create_document_store_ingestor(
    document_store_uri: str,
    document_store_collection_name: str,
    embedding_model_path: str,
) -> DocumentStoreIngestor:
    """
    Creates a `DocumentStoreIngestor` instance using the provided settings.

    Args:
        document_store_uri: URI of the document store service (or file if it's an embedded instance).
        document_store_collection_name: Name of the document store collection from which the embeddings are retrieved.
        top_k: Number of documents to retrieve at each request.
        embedding_model_path: Path of the embedding model used to generate the query embeddings.

    Returns:
        An instance of `DocumentStoreIngestor` according to the provided settings.
    """

    # First Party
    from instructlab.rag.haystack.document_store_factory import (
        create_in_memory_document_store,
    )

    return create_in_memory_document_store(
        document_store_uri=document_store_uri,
        document_store_collection_name=document_store_collection_name,
        embedding_model_path=embedding_model_path,
    )
