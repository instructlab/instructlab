# Standard
from pathlib import Path
import glob
import logging
import os

# Third Party
from haystack import Pipeline  # type: ignore
from haystack.components.converters import TextFileToDocument  # type: ignore
from haystack.components.embedders import (  # type: ignore
    SentenceTransformersDocumentEmbedder,
)
from haystack.components.preprocessors import DocumentCleaner  # type: ignore
from haystack.components.writers import DocumentWriter  # type: ignore
from milvus_haystack import MilvusDocumentStore  # type: ignore

# First Party
from instructlab.haystack.docling_splitter import DoclingDocumentSplitter
from instructlab.rag.rag_configuration import (
    document_store_configuration,
    embedder_configuration,
)

logger = logging.getLogger(__name__)


def ingest_docs(
    input_dir: str,
    document_store_config: document_store_configuration,
    embedder_config: embedder_configuration,
):
    pipeline = _create_pipeline(
        document_store_config=document_store_config,
        embedder_config=embedder_config,
    )
    _connect_components(pipeline)
    _ingest_docs(pipeline=pipeline, input_dir=input_dir)


def _create_pipeline(
    document_store_config: document_store_configuration,
    embedder_config: embedder_configuration,
) -> Pipeline:
    pipeline = Pipeline()
    pipeline.add_component(instance=_converter_component(), name="converter")
    pipeline.add_component(instance=DocumentCleaner(), name="document_cleaner")
    # TODO make the params configurable
    pipeline.add_component(
        instance=_splitter_component(embedder_config),
        name="document_splitter",
    )
    # TODO make this more generic
    pipeline.add_component(
        instance=SentenceTransformersDocumentEmbedder(
            model=embedder_config.local_model_path()
        ),
        name="document_embedder",
    )
    pipeline.add_component(
        instance=DocumentWriter(
            _document_store_component(
                document_store_type=document_store_config.type,
                document_store_uri=document_store_config.uri,
                collection_name=document_store_config.collection_name,
            )
        ),
        name="document_writer",
    )
    return pipeline


def _connect_components(pipeline):
    pipeline.connect("converter", "document_cleaner")
    pipeline.connect("document_cleaner", "document_splitter")
    pipeline.connect("document_splitter", "document_embedder")
    pipeline.connect("document_embedder", "document_writer")


def _ingest_docs(pipeline, input_dir):
    pattern = "*json"
    if Path(os.path.join(input_dir, "docling-artifacts")).exists():
        pattern = "docling-artifacts/" + pattern

    ingestion_results = pipeline.run(
        {"converter": {"sources": glob.glob(os.path.join(input_dir, pattern))}}
    )

    document_store = pipeline.get_component("document_writer").document_store
    logger.info(f"count_documents: {document_store.count_documents()}")
    logger.info(
        f"document_writer.documents_written: {ingestion_results['document_writer']['documents_written']}"
    )


def _document_store_component(document_store_type, document_store_uri, collection_name):
    if document_store_type == "milvuslite":
        document_store = MilvusDocumentStore(
            connection_args={"uri": document_store_uri},
            collection_name=collection_name,
            drop_old=True,
        )
        return document_store
    else:
        raise ValueError(f"Unmanaged document store type {document_store_type}")


def _converter_component():
    return TextFileToDocument()


def _splitter_component(embedding_model):
    return DoclingDocumentSplitter(
        embedding_model_id=embedding_model.local_model_path(),
        content_format="json",
        max_tokens=150,
    )
