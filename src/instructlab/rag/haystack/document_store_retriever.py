# SPDX-License-Identifier: Apache-2.0

# Standard
import logging

# Third Party
from haystack import Pipeline  # type: ignore

# First Party
from instructlab.rag.document_store import DocumentStoreRetriever
from instructlab.rag.haystack.component_factory import (
    create_document_store,
    create_retriever,
    create_text_embedder,
)

logger = logging.getLogger(__name__)


class HaystackDocumentStoreRetriever(DocumentStoreRetriever):
    """
    An implementation of the `DocumentStoreRetriever` interface using the Haystack framework.
    The concrete class instance is defined using an Haystack `Pipeline` to implement the `augmented_context` method.

    The pipeline is defined by the following chain of components:
    * A text embedder, receiving the user query as the input parameter.
    * A document store, where the document embeddings have been ingested.
    * A document retriever receiving the embedded query and returning the matching documents from the document store.

    The output of the `augmented_context` method is the concatenation of the matching documents.
    """

    def __init__(
        self,
        document_store_uri: str,
        document_store_collection_name: str,
        top_k: int,
        embedding_model_path: str,
    ):
        super().__init__()
        self._pipeline = _create_pipeline(
            document_store_uri=document_store_uri,
            document_store_collection_name=document_store_collection_name,
            top_k=top_k,
            embedding_model_path=embedding_model_path,
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
    document_store_uri: str,
    document_store_collection_name: str,
    top_k: int,
    embedding_model_path: str,
) -> Pipeline:
    document_store = create_document_store(
        document_store_uri=document_store_uri,
        document_store_collection_name=document_store_collection_name,
        drop_old=False,
    )
    document_retriever = create_retriever(
        top_k=top_k,
        document_store=document_store,
    )
    text_embedder = create_text_embedder(embedding_model_path=embedding_model_path)
    pipeline = Pipeline()
    pipeline.add_component("embedder", text_embedder)
    pipeline.add_component("retriever", document_retriever)
    return pipeline


def _connect_components(pipeline: Pipeline):
    pipeline.connect("embedder.embedding", "retriever.query_embedding")
