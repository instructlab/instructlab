# Standard
from pathlib import Path
import logging
import os

# Third Party
from instructlab.sdg.utils.chunkers import DocumentChunker

logger = logging.getLogger(__name__)


def transform_docs(input_dir, output_dir, model_path):
    logger.info(f"Transforming {input_dir} to {output_dir} using {model_path}")
    if not Path(output_dir).exists():
        os.makedirs(output_dir, exist_ok=True)

    source_files = _load_source_files(input_dir=input_dir)
    logger.info(f"Transforming source files {[p.name for p in source_files]}")

    tokenizer_model_name = model_path
    logger.info(f"Tokenizer model is {tokenizer_model_name}")

    chunker = DocumentChunker(
        leaf_node=_dummy_leaf_node(source_files),
        taxonomy_path=output_dir,
        output_dir=output_dir,
        tokenizer_model_name=tokenizer_model_name,
    )
    logger.info(f"Created chunker {chunker}")
    chunks = chunker.chunk_documents()
    logger.info(
        f"Created {len(chunks)} chunks from {len(source_files)} input documents"
    )


def _load_source_files(input_dir) -> list[Path]:
    return [Path(os.path.join(input_dir, f)) for f in os.listdir(input_dir)]


def _dummy_leaf_node(source_files):
    return [
        {
            "documents": ["not relevant"],
            "taxonomy_path": "not relevant",
            "filepaths": source_files,
        }
    ]
