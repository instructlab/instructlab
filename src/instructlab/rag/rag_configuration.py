# Standard
import enum
import os

# Third Party
from pydantic import BaseModel, ConfigDict, Field
import click

# First Party
from instructlab.configuration import DEFAULTS


class RagFramework(enum.Enum):
    Haystack: str = "haystack"


class DocumentStoreType(enum.Enum):
    Milvus: str = "milvus"


class DocumentStoreConfig(BaseModel):
    # model configuration
    model_config = ConfigDict(
        extra="ignore",
        protected_namespaces=(),
        use_enum_values=False,  # populate models with the value property of enums, rather than the raw enum.
    )
    type: DocumentStoreType = Field(
        default=DocumentStoreType.Milvus,
        description=f"The document store type, one of: {[type.value for type in DocumentStoreType]}",
    )
    uri: str = Field(
        default="embeddings.db",
        description="The document store URI",
    )
    collection_name: str = Field(
        default="IlabEmbeddings",
        description="The document store collection name",
    )


class EmbeddingModelConfig(BaseModel):
    # model configuration
    model_config = ConfigDict(
        extra="ignore",
        protected_namespaces=(),
        use_enum_values=True,  # populate models with the value property of enums, rather than the raw enum.
    )
    model_dir: str = Field(
        default=DEFAULTS.MODELS_DIR,
        description="The default system model location store, located in the data directory.",
    )
    model_name: str = Field(
        # default_factory=lambda: "ibm-granite/granite-embedding-278m-multilingual",
        default_factory=lambda: "sentence-transformers/all-minilm-l6-v2",
        description="The embedding model name.",
    )

    def local_model_path(self) -> str:
        if self.model_dir is None:
            click.secho(f"Missing value for field model_dir in {vars(self)}")
            raise click.exceptions.Exit(1)

        if self.model_name is None:
            click.secho(f"Missing value for field model_name in {vars(self)}")
            raise click.exceptions.Exit(1)

        return os.path.join(self.model_dir, self.model_name)


class RetrieverConfig(BaseModel):
    top_k: int = Field(
        default=10,
        description="The maximum number of `Document`s to return.",
    )
    embedding_model: EmbeddingModelConfig = Field(
        default_factory=lambda: EmbeddingModelConfig(),
        description="The embedding model.",
    )
