# Third Party
from haystack.components.writers import DocumentWriter  # type: ignore

# from haystack_integrations.document_stores.elasticsearch import (
#     ElasticsearchDocumentStore,
# )
from milvus_haystack import MilvusDocumentStore  # type: ignore

# First Party
from instructlab.rag.rag_configuration import DocumentStoreConfig


def create_milvus_document_store(config: DocumentStoreConfig):
    return DocumentWriter(
        MilvusDocumentStore(
            connection_args={"uri": config.uri},
            collection_name=config.collection_name,
            drop_old=True,
        )
    )


def create_milvus_document_retriever(config: DocumentStoreConfig):
    return MilvusDocumentStore(
        connection_args={"uri": config.uri},
        collection_name=config.collection_name,
        drop_old=False,
    )


# def create_elasticsearch_document_store(config: DocumentStoreConfig):
#     return ElasticsearchDocumentStore(hosts=config.uri)
