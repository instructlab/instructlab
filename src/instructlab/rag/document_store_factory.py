# Standard
from typing import cast
import importlib
import logging

# First Party
from instructlab.rag.document_store import DocumentStoreIngestor, DocumentStoreRetriever
from instructlab.rag.rag_configuration import (
    DocumentStoreConfig,
    DocumentStoreType,
    EmbeddingModelConfig,
    RagFramework,
    RetrieverConfig,
)

logger = logging.getLogger(__name__)


def create_document_retriever(
    framework: RagFramework,
    type: DocumentStoreType,
    document_store_config: DocumentStoreConfig,
    retriever_config: RetrieverConfig,
) -> DocumentStoreRetriever:
    try:
        module_name = f"instructlab.rag.{framework.value}.document_store_factory"
        module = importlib.import_module(module_name)

        function_name = f"create_{type.value}_document_retriever"
        function = getattr(module, function_name)

        retriever = function(
            document_store_config=document_store_config,
            retriever_config=retriever_config,
        )
        return cast(DocumentStoreRetriever, retriever)
    except ModuleNotFoundError as exc:
        logger.warning(f"Module not found {module_name}")
        raise exc
    except AttributeError as exc:
        logger.warning(f"Function {function_name} not found in module {module_name}")
        raise exc


def create_document_store_ingestor(
    framework: RagFramework,
    type: DocumentStoreType,
    document_store_config: DocumentStoreConfig,
    embedding_config: EmbeddingModelConfig,
) -> DocumentStoreIngestor:
    try:
        module_name = f"instructlab.rag.{framework.value}.document_store_factory"
        module = importlib.import_module(module_name)

        function_name = f"create_{type.value}_document_store"
        function = getattr(module, function_name)

        retriever = function(
            document_store_config=document_store_config,
            embedding_config=embedding_config,
        )
        return cast(DocumentStoreIngestor, retriever)
    except ModuleNotFoundError as exc:
        logger.warning(f"Module not found {module_name}")
        raise exc
    except AttributeError as exc:
        logger.warning(f"Function {function_name} not found in module {module_name}")
        raise exc
