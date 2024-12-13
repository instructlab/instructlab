# SPDX-License-Identifier: Apache-2.0

# Standard
import logging

# Third Party
# RAG
from haystack import Pipeline  # type: ignore
from haystack.components.embedders import (  # type: ignore
    SentenceTransformersTextEmbedder,
)
from milvus_haystack import MilvusDocumentStore  # type: ignore
from milvus_haystack.milvus_embedding_retriever import (  # type: ignore
    MilvusEmbeddingRetriever,
)

# First Party
from instructlab.rag.rag_configuration import RagConfig

logger = logging.getLogger(__name__)


_DEFAULT_RAG_PROMPT = """
Given the following information, answer the question.
Context:
{context}
Question: 
{user_query}
Answer:
"""


def rag_prompt() -> str:
    return _DEFAULT_RAG_PROMPT


def init_rag_chat_pipeline(
    rag_config: RagConfig,
):
    if rag_config.document_store.type == "milvuslite":
        document_store = MilvusDocumentStore(
            connection_args={"uri": rag_config.document_store.uri},
            collection_name=rag_config.document_store.collection_name,
            drop_old=False,
        )
        logger.debug(f"RAG document_store created {document_store}")

        document_retriever = MilvusEmbeddingRetriever(
            document_store=document_store,
            top_k=rag_config.retriever.top_k,
        )
        logger.debug(f"RAG document_retriever created {document_retriever}")

        text_embedder = SentenceTransformersTextEmbedder(
            model=rag_config.retriever.embedder.local_model_path()
        )
        logger.debug(f"RAG text_embedder created {text_embedder}")

        pipeline = Pipeline()
        pipeline.add_component("embedder", text_embedder)
        pipeline.add_component("retriever", document_retriever)
        pipeline.connect("embedder.embedding", "retriever.query_embedding")
        logger.debug(f"RAG pipeline created {pipeline}")

        return pipeline
    else:
        raise ValueError(
            f"Unmanaged document store type {rag_config.document_store.type}"
        )
