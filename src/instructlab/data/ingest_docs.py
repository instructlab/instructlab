# Standard
import glob
import logging
import os

# Third Party
from haystack import Pipeline  # type: ignore
from haystack.components.converters import TextFileToDocument
from haystack.components.embedders import (  # type: ignore
    SentenceTransformersDocumentEmbedder,
)
from haystack.components.preprocessors import DocumentCleaner  # type: ignore
from haystack.components.writers import DocumentWriter  # type: ignore
from milvus_haystack import MilvusDocumentStore  # type: ignore

# First Party
from instructlab.haystack.docling_splitter import DoclingDocumentSplitter

logger = logging.getLogger(__name__)


class VectorDbParams:
    def __init__(
        self,
        vectordb_type,
        vectordb_uri,
        vectordb_collection_name,
    ):
        self.type = vectordb_type
        self.uri = vectordb_uri
        self.collection_name = vectordb_collection_name


class EmbeddingModel:
    def __init__(
        self,
        model_dir,
        model_name,
    ):
        self.model_dir = model_dir
        self.model_name = model_name

    def validate_local_model_path(self):
        local_model_path = self.local_model_path()
        return os.path.exists(local_model_path) and os.path.isdir(local_model_path)

    def local_model_path(self) -> str:
        return str(os.path.join(self.model_dir, self.model_name))


def ingest_docs(
    input_dir: str, vectordb_params: VectorDbParams, embedding_model: EmbeddingModel
):
    pipeline = _create_pipeline(
        vectordb_params=vectordb_params,
        embedding_model=embedding_model,
    )
    _connect_components(pipeline)
    _ingest_docs(pipeline=pipeline, input_dir=input_dir)


def _create_pipeline(
    vectordb_params: VectorDbParams, embedding_model: EmbeddingModel
) -> Pipeline:
    pipeline = Pipeline()
    pipeline.add_component(instance=_converter(), name="converter")
    pipeline.add_component(instance=DocumentCleaner(), name="document_cleaner")
    # TODO make the params configurable
    pipeline.add_component(
        instance=_splitter(embedding_model),
        name="document_splitter",
    )
    # TODO make this more generic
    pipeline.add_component(
        instance=SentenceTransformersDocumentEmbedder(
            model=embedding_model.local_model_path()
        ),
        name="document_embedder",
    )
    pipeline.add_component(
        instance=DocumentWriter(
            _document_store(
                vectordb_type=vectordb_params.type,
                vectordb_uri=vectordb_params.uri,
                collection_name=vectordb_params.collection_name,
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
    ingestion_results = pipeline.run(
        {
            "converter": {
                "sources": glob.glob(os.path.join(input_dir, "docling-artifacts/*json"))
            }
        }
    )

    document_store = pipeline.get_component("document_writer").document_store
    logger.info(f"count_documents: {document_store.count_documents()}")
    logger.info(
        f"document_writer.documents_written: {ingestion_results['document_writer']['documents_written']}"
    )


def _document_store(vectordb_type, vectordb_uri, collection_name):
    if vectordb_type == "milvuslite":
        document_store = MilvusDocumentStore(
            connection_args={"uri": vectordb_uri},
            collection_name=collection_name,
            drop_old=True,
        )
        return document_store
    else:
        raise ValueError(f"Unmanaged vector db type {vectordb_type}")


def _converter():
    return TextFileToDocument()


def _splitter(embedding_model):
    return DoclingDocumentSplitter(
        embedding_model_id=embedding_model.local_model_path(),
        content_format="json",
        max_tokens=150,
    )
