# Standard
import enum

# Third Party
import pytest

# First Party
from src.instructlab.rag.document_store_factory import (
    create_document_retriever,
    create_document_store,
)
from src.instructlab.rag.rag_configuration import (
    DocumentStoreConfig,
    DocumentStoreType,
    RagFramework,
)


def test_document_retriever_for_haystack_milvus():
    config = DocumentStoreConfig(
        type=DocumentStoreType.Milvus, uri="milvus.db", collection_name="default"
    )
    retriever = create_document_retriever(
        framework=RagFramework.Haystack, type=DocumentStoreType.Milvus, config=config
    )

    assert retriever is not None
    assert type(retriever).__module__ == "milvus_haystack.document_store"
    assert type(retriever).__name__ == "MilvusDocumentStore"


def test_document_writer_for_haystack_milvus():
    config = DocumentStoreConfig(
        type=DocumentStoreType.Milvus, uri="milvus.db", collection_name="default"
    )
    writer = create_document_store(
        framework=RagFramework.Haystack, type=DocumentStoreType.Milvus, config=config
    )

    assert writer is not None
    assert type(writer).__module__ == "haystack.components.writers.document_writer"
    assert type(writer).__name__ == "DocumentWriter"


def test_document_retriever_for_unmanaged_framework():
    ExtendedFramework = enum.Enum(
        "ExtendedFramework",
        {**{item.name: item.value for item in RagFramework}, "UNMANAGED": "unmanaged"},
    )

    config = DocumentStoreConfig(
        type=DocumentStoreType.Milvus, uri="milvus.db", collection_name="default"
    )
    with pytest.raises(ModuleNotFoundError):
        create_document_store(
            framework=ExtendedFramework.UNMANAGED,
            type=DocumentStoreType.Milvus,
            config=config,
        )

    ExtendedDocumentStoreType = enum.Enum(
        "ExtendedDocumentStoreType",
        {
            **{item.name: item.value for item in DocumentStoreType},
            "UNMANAGED": "unmanaged",
        },
    )
    with pytest.raises(AttributeError):
        create_document_store(
            framework=RagFramework.Haystack,
            type=ExtendedDocumentStoreType.UNMANAGED,
            config=config,
        )
