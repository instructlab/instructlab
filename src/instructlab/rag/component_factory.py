# Third Party
from haystack.components.embedders import (  # type: ignore
    SentenceTransformersTextEmbedder,
)
from milvus_haystack import MilvusDocumentStore  # type: ignore
from milvus_haystack.milvus_embedding_retriever import (  # type: ignore
    MilvusEmbeddingRetriever,
)

# First Party
from instructlab.rag.rag_configuration import _document_store, _retriever


def create_document_store(document_store_config: _document_store, drop_old: bool):
    if document_store_config.type == "milvuslite":
        return MilvusDocumentStore(
            connection_args={"uri": document_store_config.uri},
            collection_name=document_store_config.collection_name,
            drop_old=drop_old,
        )
    else:
        raise ValueError(f"Unmanaged document store type {document_store_config.type}")


def create_retriever(
    document_store_config: _document_store,
    retriever_config: _retriever,
    document_store: MilvusDocumentStore,
):
    if document_store_config.type == "milvuslite":
        return MilvusEmbeddingRetriever(
            document_store=document_store,
            top_k=retriever_config.top_k,
        )
    else:
        raise ValueError(f"Unmanaged document store type {document_store_config.type}")


def create_embedder(retriever_config: _retriever):
    return SentenceTransformersTextEmbedder(
        model=retriever_config.embedder.local_model_path()
    )
