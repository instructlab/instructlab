# SPDX-License-Identifier: Apache-2.0

# Standard
import logging

# Third Party
from haystack import Pipeline  # type: ignore

# First Party
from instructlab.rag.component_factory import (
    create_document_store,
    create_embedder,
    create_retriever,
)
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
    document_store = create_document_store(rag_config.document_store, drop_old=False)
    logger.debug(f"RAG document_store created {document_store}")

    document_retriever = create_retriever(
        document_store_config=rag_config.document_store,
        retriever_config=rag_config.retriever,
        document_store=document_store,
    )
    logger.debug(f"RAG document_retriever created {document_retriever}")

    text_embedder = create_embedder(rag_config.retriever)
    logger.debug(f"RAG text_embedder created {text_embedder}")

    pipeline = Pipeline()
    pipeline.add_component("embedder", text_embedder)
    pipeline.add_component("retriever", document_retriever)
    pipeline.connect("embedder.embedding", "retriever.query_embedding")
    logger.debug(f"RAG pipeline created {pipeline}")

    return pipeline


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
