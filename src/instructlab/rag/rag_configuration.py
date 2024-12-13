# Standard
from typing import Any, Optional
import logging
import os

# Third Party
from pydantic import BaseModel, ConfigDict
import click

# First Party
from instructlab.configuration import DEFAULTS

logger = logging.getLogger(__name__)


def rag_options(command):
    """Wrapper to apply common options."""
    command = click.option(
        "--rag",
        "is_rag",
        is_flag=True,
        envvar="ILAB_RAG",
        help="To enable Retrieval-Augmented Generation",
    )(command)
    command = click.option(
        "--document-store-type",
        default="milvuslite",
        envvar="ILAB_DOCUMENT_STORE_TYPE",
        type=click.STRING,
        help="The document store type, one of: `milvuslite`, `milvus`.",
    )(command)
    command = click.option(
        "--document-store-uri",
        default="embeddings.db",
        envvar="ILAB_DOCUMENT_STORE_URI",
        type=click.STRING,
        help="The document store URI",
    )(command)
    command = click.option(
        "--document-store-collection-name",
        default="Ilab",
        envvar="ILAB_DOCUMENT_STORE_COLLECTION_NAME",
        type=click.STRING,
        help="The document store collection name",
    )(command)
    command = click.option(
        "--retriever-top-k",
        default=10,
        envvar="ILAB_RETRIEVER_TOP_K",
        type=click.INT,
    )(command)
    command = click.option(
        "--retriever-embedder-model-dir",
        default=lambda: DEFAULTS.MODELS_DIR,
        envvar="ILAB_EMBEDDER_MODEL_DIR",
        show_default="The default system model location store, located in the data directory.",
        help="Base directories where models are stored.",
    )(command)
    command = click.option(
        "--retriever-embedder-model-name",
        default="sentence-transformers/all-minilm-l6-v2",
        envvar="ILAB_EMBEDDER_MODEL_NAME",
        type=click.STRING,
        help="The embedding model name",
    )(command)
    return command


class _document_store(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: Optional[str] = None
    uri: Optional[str] = None
    collection_name: Optional[str] = None


class _embedder(BaseModel):
    model_dir: Optional[str] = None
    model_name: Optional[str] = None

    def validate_local_model_path(self):
        local_model_path = self.local_model_path()
        return os.path.exists(local_model_path) and os.path.isdir(local_model_path)

    def local_model_path(self) -> str:
        return str(os.path.join(self.model_dir, self.model_name))


class _retriever(BaseModel):
    top_k: Optional[int] = None
    embedder: Optional[_embedder] = _embedder()


class RagConfig:
    def __init__(self, rag_config: dict[str, Any], **kwargs):
        logger.debug(f"init from {rag_config}")
        logger.debug(f"init from {kwargs}")
        self.enabled = rag_config.get("enable") or kwargs.get("is_rag")
        self.document_store = _document_store(
            **(rag_config.get("document_store") or {})
        )
        self.retriever = _retriever(**(rag_config.get("retriever") or {}))

        logger.debug(f"Before injecting config: {vars(self)}")
        init_from_flags(model=self.document_store, **kwargs)
        init_from_flags(model=self.retriever, **kwargs)
        logger.debug(f"After injecting config: {vars(self)}")


def no_rag_config() -> RagConfig:
    rag_config = RagConfig()
    rag_config.enabled = False
    return rag_config


def init_from_flags(model: BaseModel, prefix="", **kwargs):
    model_name = model.__class__.__name__
    if model_name.startswith("_"):
        model_name = model_name[1:] + "_"
    model_name = prefix + model_name
    logger.debug(f"model_name is {model_name}")
    for key, value in kwargs.items():
        if value is not None:
            logger.debug(f"key is {key}")
            if key.startswith(model_name):
                attr_name = key.replace(model_name, "")
                if hasattr(model, attr_name):
                    logger.debug(f"Overriding from flag {key}")
                    setattr(model, attr_name, value)

    prefix = model_name
    for _, value in vars(model).items():
        if isinstance(value, BaseModel):
            init_from_flags(model=value, prefix=prefix, **kwargs)
