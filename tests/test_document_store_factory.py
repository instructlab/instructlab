# Standard
import enum
import os
import shutil
import tempfile

# Third Party
from pymilvus import MilvusClient
import pytest

# First Party
from instructlab.rag.document_store import DocumentStoreIngestor, DocumentStoreRetriever
from instructlab.rag.document_store_factory import (
    create_document_retriever,
    create_document_store_ingestor,
)
from instructlab.rag.rag_configuration import (
    DocumentStoreConfig,
    DocumentStoreType,
    EmbeddingModelConfig,
    RagFramework,
    RetrieverConfig,
)


def test_document_ingestor_for_haystack_milvus():
    with tempfile.TemporaryDirectory() as temp_dir:
        document_store_config = DocumentStoreConfig(
            type=DocumentStoreType.Milvus,
            uri=os.path.join(temp_dir, "milvus.db"),
            collection_name="default",
        )

        embedding_config = EmbeddingModelConfig()
        ingestor: DocumentStoreIngestor = create_document_store_ingestor(
            framework=RagFramework.Haystack,
            type=DocumentStoreType.Milvus,
            document_store_config=document_store_config,
            embedding_config=embedding_config,
        )

        assert ingestor is not None
        assert isinstance(ingestor, DocumentStoreIngestor) is True
        assert (
            type(ingestor).__module__
            == "instructlab.rag.haystack.document_store_ingestor"
        )
        assert (
            type(ingestor).__module__
            == "instructlab.rag.haystack.document_store_ingestor"
        )
        assert type(ingestor).__name__ == "HaystackDocumentStoreIngestor"


def test_document_retriever_for_haystack_milvus():
    with tempfile.TemporaryDirectory() as temp_dir:
        document_store_config = DocumentStoreConfig(
            type=DocumentStoreType.Milvus,
            uri=os.path.join(temp_dir, "milvus.db"),
            collection_name="default",
        )
        retriever_config = RetrieverConfig()
        retriever: DocumentStoreRetriever = create_document_retriever(
            framework=RagFramework.Haystack,
            type=DocumentStoreType.Milvus,
            document_store_config=document_store_config,
            retriever_config=retriever_config,
        )

        assert retriever is not None
        assert isinstance(retriever, DocumentStoreRetriever) is True
        assert (
            type(retriever).__module__
            == "instructlab.rag.haystack.document_store_retriever"
        )
        assert type(retriever).__name__ == "HaystackDocumentStoreRetriever"


def test_document_store_ingest_and_retrieve_for_haystack_milvus():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Ingest docus from a test folder
        document_store_config = DocumentStoreConfig(
            type=DocumentStoreType.Milvus,
            uri=os.path.join(temp_dir, "ingest.db"),
            collection_name="default",
        )
        embedding_config = EmbeddingModelConfig()
        ingestor: DocumentStoreIngestor = create_document_store_ingestor(
            framework=RagFramework.Haystack,
            type=DocumentStoreType.Milvus,
            document_store_config=document_store_config,
            embedding_config=embedding_config,
        )

        input_dir = "tests/testdata/temp_datasets_documents"
        result, count = ingestor.ingest_documents(input_dir)

        assert result is True
        assert count > 0

        # Validate document store collection
        client = MilvusClient(document_store_config.uri)
        collections = client.list_collections()
        assert len(collections) == 1
        assert collections[0] == document_store_config.collection_name
        stats = client.get_collection_stats(document_store_config.collection_name)
        assert stats is not None
        assert type(stats) is dict
        assert "row_count" in stats
        assert stats["row_count"] == count
        client.close()

        # Run a retriever session
        # Copy db file to avoid concurrent access issues
        new_file = os.path.join(temp_dir, "query.db")
        shutil.copy(document_store_config.uri, new_file)
        document_store_config = DocumentStoreConfig(
            type=DocumentStoreType.Milvus,
            uri=new_file,
            collection_name="default",
        )
        retriever_config = RetrieverConfig(embedding_config=embedding_config)
        retriever: DocumentStoreRetriever = create_document_retriever(
            framework=RagFramework.Haystack,
            type=DocumentStoreType.Milvus,
            document_store_config=document_store_config,
            retriever_config=retriever_config,
        )

        context = retriever.augmented_context(user_query="What is knowledge")
        assert context is not None
        assert len(context) > 0
        assert "familiarity with individuals" in context


def test_document_retriever_for_unmanaged_framework():
    ExtendedFramework = enum.Enum(
        "ExtendedFramework",
        {**{item.name: item.value for item in RagFramework}, "UNMANAGED": "unmanaged"},
    )

    document_store_config = DocumentStoreConfig(
        type=DocumentStoreType.Milvus, uri="milvus.db", collection_name="default"
    )
    retriever_config = RetrieverConfig()
    with pytest.raises(ModuleNotFoundError):
        create_document_retriever(
            framework=ExtendedFramework.UNMANAGED,
            type=DocumentStoreType.Milvus,
            document_store_config=document_store_config,
            retriever_config=retriever_config,
        )

    ExtendedDocumentStoreType = enum.Enum(
        "ExtendedDocumentStoreType",
        {
            **{item.name: item.value for item in DocumentStoreType},
            "UNMANAGED": "unmanaged",
        },
    )
    with pytest.raises(AttributeError):
        create_document_retriever(
            framework=RagFramework.Haystack,
            type=ExtendedDocumentStoreType.UNMANAGED,
            document_store_config=document_store_config,
            retriever_config=retriever_config,
        )
