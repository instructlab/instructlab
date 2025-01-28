# Third Party

# from haystack_integrations.document_stores.elasticsearch import (
#     ElasticsearchDocumentStore,
# )
# Standard
import logging

# First Party
from instructlab.rag.document_store import DocumentStoreIngestor, DocumentStoreRetriever
from instructlab.rag.haystack.document_store_ingestor import (
    HaystackDocumentStoreIngestor,
)
from instructlab.rag.haystack.document_store_retriever import (
    HaystackDocumentStoreRetriever,
)

logger = logging.getLogger(__name__)


def create_in_memory_document_store(
    document_store_uri: str,
    document_store_collection_name: str,
    embedding_model_path: str,
) -> DocumentStoreIngestor:
    """
    Creates a `DocumentStoreIngestor` based on Haystack components.
    """

    return HaystackDocumentStoreIngestor(
        document_store_uri=document_store_uri,
        document_store_collection_name=document_store_collection_name,
        embedding_model_path=embedding_model_path,
    )


def create_in_memory_document_retriever(
    document_store_uri: str,
    document_store_collection_name: str,
    top_k: int,
    embedding_model_path: str,
) -> DocumentStoreRetriever:
    """
    Creates a `DocumentStoreRetriever` based on Haystack components.
    """

    return HaystackDocumentStoreRetriever(
        document_store_uri=document_store_uri,
        document_store_collection_name=document_store_collection_name,
        top_k=top_k,
        embedding_model_path=embedding_model_path,
    )
