# Third Party
from haystack.components.converters import TextFileToDocument  # type: ignore
from haystack.components.embedders import (  # type: ignore
    SentenceTransformersTextEmbedder,
)
from milvus_haystack import MilvusDocumentStore  # type: ignore
from milvus_haystack.milvus_embedding_retriever import (  # type: ignore
    MilvusEmbeddingRetriever,
)
import click

# First Party
from instructlab.rag.haystack.components.document_splitter import (
    DoclingDocumentSplitter,
)
from instructlab.rag.rag_configuration import (
    DocumentStoreConfig,
    DocumentStoreType,
    EmbeddingModelConfig,
    RetrieverConfig,
)


def create_document_store(document_store_config: DocumentStoreConfig, drop_old: bool):
    if document_store_config.type == DocumentStoreType.Milvus:
        return MilvusDocumentStore(
            connection_args={"uri": document_store_config.uri},
            collection_name=document_store_config.collection_name,
            drop_old=drop_old,
        )
    else:
        click.secho(f"Unmanaged document store type {document_store_config.type}")
        raise click.exceptions.Exit(1)


def create_retriever(
    document_store_config: DocumentStoreConfig,
    retriever_config: RetrieverConfig,
    document_store: MilvusDocumentStore,
):
    if document_store_config.type == DocumentStoreType.Milvus:
        return MilvusEmbeddingRetriever(
            document_store=document_store,
            top_k=retriever_config.top_k,
        )
    else:
        raise ValueError(f"Unmanaged document store type {document_store_config.type}")


def create_embedder(retriever_config: RetrieverConfig):
    if retriever_config.embedding_model is None:
        raise ValueError(
            f"Missing value for field embedding_model in {vars(retriever_config)}"
        )

    return SentenceTransformersTextEmbedder(
        model=retriever_config.embedding_model.local_model_path()
    )


def create_converter():
    return TextFileToDocument()


def create_splitter(embedding_config: EmbeddingModelConfig):
    return DoclingDocumentSplitter(
        embedding_model_id=embedding_config.local_model_path(),
        content_format="json",
        max_tokens=150,
    )
