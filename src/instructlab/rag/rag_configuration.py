# Standard
import enum

# Third Party
from pydantic import BaseModel, ConfigDict, Field


class RagFramework(enum.Enum):
    Haystack: str = "haystack"


class DocumentStoreType(enum.Enum):
    Milvus: str = "milvus"


class DocumentStoreConfig(BaseModel):
    # model configuration
    model_config = ConfigDict(
        extra="ignore",
        protected_namespaces=(),
        use_enum_values=True,  # populate models with the value property of enums, rather than the raw enum.
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
