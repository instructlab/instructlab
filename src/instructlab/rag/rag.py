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


def _init_rag_chat_pipeline(
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


class RagHandler:
    def __init__(self, rag_config: RagConfig):
        self._rag_config = rag_config
        self.rag_pipeline = None
        self.rag_prompt = rag_prompt()

    def is_enabled(self) -> bool:
        return self._rag_config.enabled

    def toggle_state(self) -> bool:
        self._rag_config.enabled = not self._rag_config.enabled

    def augment_user_query(self, user_query: str) -> str:
        if self.rag_pipeline is None:
            self.rag_pipeline = _init_rag_chat_pipeline(
                rag_config=self._rag_config,
            )

        retrieval_results = self.rag_pipeline.run(
            {
                "embedder": {"text": user_query},
            }
        )
        context = "\n".join(
            [doc.content for doc in retrieval_results["retriever"]["documents"]]
        )

        logger.debug("-" * 10)
        logger.debug(f"RAG context is {context}")
        logger.debug("-" * 10)

        updated_user_query = self.rag_prompt.format(
            context=context, user_query=user_query
        )
        logger.debug(f"Updated user query is {updated_user_query}")
        return updated_user_query
