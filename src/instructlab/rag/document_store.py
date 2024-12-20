# Standard
from abc import ABC, abstractmethod

"""
A module to define interfaces to manage document stores:
- Ingest documents in it. This typically involves creating the document embeddings through a configured embedding model.
- Query the stored documents for an augmented RAG context. Advanced retrieval techniques can be used to improve the reliabiliyty of
the generated context.
"""


class DocumentStoreRetriever(ABC):
    @abstractmethod
    def augmented_context(self, user_query: str) -> str:
        """
        Retrieve documents from the actual store matching the given `user_query` and compute the augmented context to be used
        in a RAG chat pipeline.

        Params:
          user_query: The original user query.
        Returns:
          str: The augmented context to use in a RAG chat.
        """


class DocumentStoreIngestor(ABC):
    @abstractmethod
    def ingest_documents(self, input_dir) -> tuple[bool, int]:
        """
        Ingest documents from the `input_dir` location to the document store.
        Params:
          input_dir: The folder containing user documents (exploration is not recursive). The assumption is that these documents have been
          processed using instructlab-sdg processing functions, based on docling, and are in JSON format.
        Returns:
          bool: True if the ingestion succeeded, False otherwise.
          int: The number of documents written to the document store.
        """
