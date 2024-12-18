# Standard
from abc import ABC, abstractmethod
from typing import Any, Union
import importlib
import logging

# First Party
from src.instructlab.rag.rag_configuration import (
    DocumentStoreConfig,
    DocumentStoreType,
    RagFramework,
)

logger = logging.getLogger(__name__)


class DocumentStoreRetriever(ABC):
    @abstractmethod
    def query(self, *args, **kwargs) -> list[Any]:
        """
        Retrieves `Document`s from the actual store.

        Returns a list of retrieved documents.
        """


class DocumentStoreWriter(ABC):
    @abstractmethod
    def write_documents(self, *args, **kwargs) -> Union[int, Any]:
        """
        Writes `Document`s to the actual store.
        Returns one of:
          The Number of documents written to the document store.
          An implementation specific result.
        """


def create_document_retriever(
    framework: RagFramework, type: DocumentStoreType, config: DocumentStoreConfig
):
    module_name = f"instructlab.rag.{framework.value}.document_store"
    module = importlib.import_module(module_name)

    function_name = f"create_{type.value}_document_retriever"
    function = getattr(module, function_name)

    retriever = function(config)
    return retriever


def create_document_store(
    framework: RagFramework, type: DocumentStoreType, config: DocumentStoreConfig
):
    try:
        module_name = f"instructlab.rag.{framework.value}.document_store"
        module = importlib.import_module(module_name)

        function_name = f"create_{type.value}_document_store"
        function = getattr(module, function_name)

        retriever = function(config)
        return retriever
    except ModuleNotFoundError as exc:
        logger.warning(f"Module not found {module_name}")
        raise exc
    except AttributeError as exc:
        logger.warning(f"Function {function_name} not found in module {module_name}")
        raise exc
