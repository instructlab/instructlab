# Standard
from pathlib import Path
import logging

# Third Party
from haystack import Pipeline
from haystack.components.converters import JSONConverter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.joiners import DocumentJoiner
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from milvus_haystack import MilvusDocumentStore

logger = logging.getLogger(__name__)


def ingest_docs(output_dir, vectordb_type, vectordb_uri, embedding_model):
    pipeline = _create_pipeline(
        vectordb_type=vectordb_type,
        vectordb_uri=vectordb_uri,
        embedding_model=embedding_model,
    )
    _connect_components(pipeline)
    _ingest_docs(pipeline=pipeline, output_dir=output_dir)


def _create_pipeline(vectordb_type, vectordb_uri, embedding_model) -> Pipeline:
    pipeline = Pipeline()
    pipeline.add_component(instance=_converter(), name="converter")
    pipeline.add_component(instance=DocumentJoiner(), name="document_joiner")
    pipeline.add_component(instance=DocumentCleaner(), name="document_cleaner")
    # TODO make the params configurable
    pipeline.add_component(
        instance=DocumentSplitter(split_by="word", split_length=150, split_overlap=50),
        name="document_splitter",
    )
    # TODO make this more generic
    pipeline.add_component(
        instance=SentenceTransformersDocumentEmbedder(model=embedding_model),
        name="document_embedder",
    )
    pipeline.add_component(
        instance=DocumentWriter(
            _document_store(vectordb_type=vectordb_type, vectordb_uri=vectordb_uri)
        ),
        name="document_writer",
    )
    return pipeline


def _connect_components(pipeline):
    pipeline.connect("converter", "document_joiner")
    pipeline.connect("document_joiner", "document_cleaner")
    pipeline.connect("document_cleaner", "document_splitter")
    pipeline.connect("document_splitter", "document_embedder")
    pipeline.connect("document_embedder", "document_writer")


def _ingest_docs(pipeline, output_dir):
    ingestion_results = pipeline.run(
        {"converter": {"sources": list(Path(output_dir).glob("**/*"))}}
    )

    document_store = pipeline.get_component("document_writer").document_store
    logger.info(f"count_documents: {document_store.count_documents()}")
    logger.info(
        f"document_writer.documents_written: {ingestion_results['document_writer']['documents_written']}"
    )


def _document_store(vectordb_type, vectordb_uri):
    if vectordb_type == "milvuslite":
        docs_collection_name = "UserDocs"
        document_store = MilvusDocumentStore(
            connection_args={"uri": vectordb_uri},
            collection_name=docs_collection_name,
            drop_old=True,
        )
        return document_store
    else:
        raise ValueError(f"Unmanaged vector db type {vectordb_type}")


def _converter():
    jq_expr = '.["main-text"][]'
    json_converter = JSONConverter(
        jq_schema=jq_expr, content_key="text", extra_meta_fields={"type", "name"}
    )
    return json_converter
