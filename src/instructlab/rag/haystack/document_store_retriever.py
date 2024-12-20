# SPDX-License-Identifier: Apache-2.0

# Standard
import logging

# Third Party
from haystack import Pipeline  # type: ignore

# First Party
from instructlab.rag.document_store import DocumentStoreRetriever
from instructlab.rag.haystack.component_factory import (
    create_document_store,
    create_embedder,
    create_retriever,
)
from instructlab.rag.rag_configuration import DocumentStoreConfig, RetrieverConfig

logger = logging.getLogger(__name__)


class HaystackDocumentStoreRetriever(DocumentStoreRetriever):
    def __init__(
        self,
        document_store_config: DocumentStoreConfig,
        retriever_config: RetrieverConfig,
    ):
        super().__init__()
        self.document_store_config = document_store_config
        self.retriever_config = retriever_config
        self._pipeline = _create_pipeline(
            document_store_config=document_store_config,
            retriever_config=retriever_config,
        )
        _connect_components(self._pipeline)

    def augmented_context(self, user_query: str) -> str:
        results = self._pipeline.run(
            {
                "embedder": {"text": user_query},
            }
        )
        context = "\n".join([doc.content for doc in results["retriever"]["documents"]])

        logger.debug("-" * 10)
        logger.debug(f"RAG context is {context}")
        logger.debug("-" * 10)

        return context


def _create_pipeline(
    document_store_config: DocumentStoreConfig, retriever_config: RetrieverConfig
) -> Pipeline:
    document_store = create_document_store(document_store_config, drop_old=False)
    document_retriever = create_retriever(
        document_store_config=document_store_config,
        retriever_config=retriever_config,
        document_store=document_store,
    )
    text_embedder = create_embedder(retriever_config)
    pipeline = Pipeline()
    pipeline.add_component("embedder", text_embedder)
    pipeline.add_component("retriever", document_retriever)
    return pipeline


def _connect_components(pipeline: Pipeline):
    pipeline.connect("embedder.embedding", "retriever.query_embedding")
