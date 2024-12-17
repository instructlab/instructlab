# Third Party
from haystack.components.embedders import (  # type: ignore
    SentenceTransformersTextEmbedder,
)
from milvus_haystack import MilvusDocumentStore  # type: ignore
from milvus_haystack.milvus_embedding_retriever import (  # type: ignore
    MilvusEmbeddingRetriever,
)

# First Party
from instructlab.rag.rag_configuration import (
    _retriever_configuration,
    document_store_configuration,
)


def create_document_store(
    document_store_config: document_store_configuration, drop_old: bool
):
    if document_store_config.type == "milvuslite":
        return MilvusDocumentStore(
            connection_args={"uri": document_store_config.uri},
            collection_name=document_store_config.collection_name,
            drop_old=drop_old,
        )
    else:
        raise ValueError(f"Unmanaged document store type {document_store_config.type}")


def create_retriever(
    document_store_config: document_store_configuration,
    retriever_config: _retriever_configuration,
    document_store: MilvusDocumentStore,
):
    if document_store_config.type == "milvuslite":
        return MilvusEmbeddingRetriever(
            document_store=document_store,
            top_k=retriever_config.top_k,
        )
    else:
        raise ValueError(f"Unmanaged document store type {document_store_config.type}")


def create_embedder(retriever_config: _retriever_configuration):
    if retriever_config.embedder is None:
        raise ValueError(
            f"Missing value for field embedder in {vars(retriever_config)}"
        )

    return SentenceTransformersTextEmbedder(
        model=retriever_config.embedder.local_model_path()
    )
