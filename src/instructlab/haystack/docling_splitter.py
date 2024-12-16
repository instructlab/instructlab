# Standard
from typing import Any, Dict, List, cast
import logging

# Third Party
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.types import DoclingDocument
from docling_core.types.legacy_doc.document import (
    ExportedCCSDocument as LegacyDoclingDocument,
)
from docling_core.utils.legacy import legacy_to_docling_document
from haystack import Document, component  # type: ignore
from haystack.core.serialization import (  # type: ignore
    default_from_dict,
    default_to_dict,
)
from pydantic_core._pydantic_core import ValidationError

logger = logging.getLogger(__name__)


@component
class DoclingDocumentSplitter:
    SUPPORTED_CONTENT_FORMATS = ["json"]

    def __init__(self, embedding_model_id=None, content_format=None, max_tokens=None):
        self.__chunker = HybridChunker(
            tokenizer=embedding_model_id, max_tokens=max_tokens
        )
        self.__embedding_model_id = embedding_model_id

        if content_format not in self.SUPPORTED_CONTENT_FORMATS:
            raise ValueError(
                f"Only the following input formats are currently supported: {self.SUPPORTED_CONTENT_FORMATS}."
            )
        self.__content_format = content_format

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        if not isinstance(documents, list) or (
            documents and not isinstance(documents[0], Document)
        ):
            raise TypeError(
                "DoclingDocumentSplitter expects a List of Documents as input."
            )

        split_docs = []
        for doc in documents:
            if doc.content is None:
                raise ValueError(f"Missing content for document ID {doc.id}.")

            chunks = self._split_with_docling(doc.meta["file_path"], doc.content)
            current_split_docs = [Document(content=chunk) for chunk in chunks]
            split_docs.extend(current_split_docs)

        return {"documents": split_docs}

    def _split_with_docling(self, file_path: str, text: str) -> List[str]:
        if self.__content_format == "json":
            try:
                # We expect the JSON coming from instructlab-sdg, so in docling "legacy" schema
                # See this note about the content that will not be preserved in the transformation:
                # https://github.com/DS4SD/docling-core/blob/3f631f06277a2a7301c4c7a4e45792242512ce11/docling_core/utils/legacy.py#L352
                legacy_document: LegacyDoclingDocument = (
                    LegacyDoclingDocument.model_validate_json(text)
                )
                document = legacy_to_docling_document(legacy_document)
            except ValidationError as e:
                logger.warning(
                    f"Expected {file_path} to be in legacy docling format, but schema validation failed: {e}\n"
                    + "Tring the updated schema instead."
                )
                try:
                    document = DoclingDocument.model_validate_json(text)
                except ValidationError as e:
                    logger.error(
                        f"Expected {file_path} to be in legacy docling format, but schema validation failed: {e}"
                    )
                    raise e

        else:
            raise ValueError(f"Unexpected content format {self.__content_format}")

        chunk_iter = self.__chunker.chunk(dl_doc=document)
        chunks = list(chunk_iter)
        return [self.__chunker.serialize(chunk=chunk) for chunk in chunks]

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.
        """
        return default_to_dict(  # type: ignore[no-any-return]
            self,
            embedding_model_id=self.__embedding_model_id,
            content_format=self.__content_format,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DoclingDocumentSplitter":
        """
        Deserializes the component from a dictionary.
        """
        return cast("DoclingDocumentSplitter", default_from_dict(cls, data))
