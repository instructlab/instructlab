# Third Party

# from haystack_integrations.document_stores.elasticsearch import (
#     ElasticsearchDocumentStore,
# )
# Standard
import logging

# First Party
from instructlab.rag.document_store import DocumentStoreIngestor
from instructlab.rag.haystack.document_store_ingestor import (
    HaystackDocumentStoreIngestor,
)
from instructlab.rag.haystack.document_store_retriever import (
    HaystackDocumentStoreRetriever,
)
from instructlab.rag.rag_configuration import (
    DocumentStoreConfig,
    EmbeddingModelConfig,
    RetrieverConfig,
)

logger = logging.getLogger(__name__)


def create_milvus_document_store(
    document_store_config: DocumentStoreConfig, embedding_config: EmbeddingModelConfig
) -> DocumentStoreIngestor:
    return HaystackDocumentStoreIngestor(
        document_store_config=document_store_config, embedding_config=embedding_config
    )


def create_milvus_document_retriever(
    document_store_config: DocumentStoreConfig, retriever_config: RetrieverConfig
):
    return HaystackDocumentStoreRetriever(
        document_store_config=document_store_config, retriever_config=retriever_config
    )


# def create_elasticsearch_document_store(config: DocumentStoreConfig):
#     return ElasticsearchDocumentStore(hosts=config.uri)
