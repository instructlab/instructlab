# Standard
from pathlib import Path
import glob
import logging
import os

# Third Party
from haystack import Pipeline  # type: ignore
from haystack.components.preprocessors import DocumentCleaner  # type: ignore

# First Party
from instructlab.rag.document_store import DocumentStoreIngestor
from instructlab.rag.haystack.component_factory import (
    create_converter,
    create_document_embedder,
    create_document_writer,
    create_splitter,
)

logger = logging.getLogger(__name__)


class HaystackDocumentStoreIngestor(DocumentStoreIngestor):
    """
    An implementation of the `DocumentStoreIngestor` interface using the Haystack framework.
    The concrete class instance is defined using an Haystack `Pipeline` to implement the `ingest_documents` method.

    The pipeline is defined by the following chain of components:
    * A document converter, receiving the user document to generate the Haystack `Document`.
    * A document cleaner, to remove unneeded text like extra whitespaces and empty lines.
    * A document splitter to generate smaller chunks of the original documents.
    * A document embedder to calculates document embeddings using the configured embedding model.
    * A document store, where the document embeddings are ingested.
    * A document writer to load vector embeddings to the document store.

    The output of the `ingest_documents` method is tuple with the completion status and the number
    of documents written to the document store.
    """

    def __init__(
        self,
        document_store_uri: str,
        document_store_collection_name: str,
        embedding_model_path: str,
    ):
        super().__init__()
        self.document_store_uri = document_store_uri
        self._pipeline = _create_pipeline(
            document_store_uri=document_store_uri,
            document_store_collection_name=document_store_collection_name,
            embedding_model_path=embedding_model_path,
        )
        _connect_components(self._pipeline)

    def ingest_documents(self, input_dir: str) -> tuple[bool, int]:
        pattern = "*.json"
        if Path(os.path.join(input_dir, "docling-artifacts")).exists():
            pattern = "docling-artifacts/" + pattern

        try:
            self._pipeline.run(
                {"converter": {"sources": glob.glob(os.path.join(input_dir, pattern))}}
            )
            document_store = self._pipeline.get_component(
                "document_writer"
            ).document_store
            logger.info(f"count_documents: {document_store.count_documents()}")

            # Final step required for InMemory document store
            document_store.save_to_disk(self.document_store_uri)
            logger.info(f"Saved document store as: {self.document_store_uri}")
            return True, document_store.count_documents()
        except Exception as e:
            logger.error(f"Ingestion attempt failed: {e}")
            return False, -1


def _create_pipeline(
    document_store_uri: str,
    document_store_collection_name: str,
    embedding_model_path: str,
) -> Pipeline:
    pipeline = Pipeline()
    pipeline.add_component(instance=create_converter(), name="converter")
    pipeline.add_component(instance=DocumentCleaner(), name="document_cleaner")
    # TODO make the params configurable
    pipeline.add_component(
        instance=create_splitter(embedding_model_path=embedding_model_path),
        name="document_splitter",
    )
    # TODO make this more generic
    pipeline.add_component(
        instance=create_document_embedder(embedding_model_path=embedding_model_path),
        name="document_embedder",
    )
    pipeline.add_component(
        instance=create_document_writer(
            document_store_uri=document_store_uri,
            document_store_collection_name=document_store_collection_name,
        ),
        name="document_writer",
    )
    return pipeline


def _connect_components(pipeline: Pipeline):
    pipeline.connect("converter", "document_cleaner")
    pipeline.connect("document_cleaner", "document_splitter")
    pipeline.connect("document_splitter", "document_embedder")
    pipeline.connect("document_embedder", "document_writer")
