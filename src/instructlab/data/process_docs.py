# Standard
from pathlib import Path
from typing import Iterable
import json
import logging
import os
import tempfile
import time

# Third Party
from docling.datamodel.base_models import ConversionStatus  # type: ignore
from docling.datamodel.document import ConversionResult  # type: ignore
from docling.document_converter import DocumentConverter  # type: ignore

# First Party
from instructlab.data.taxonomy_utils import lookup_knowledge_files
from instructlab.utils import clear_directory

logger = logging.getLogger(__name__)


def process_docs_from_taxonomy(taxonomy_path, taxonomy_base, output_dir):
    # Taxonomy navigation strategy:
    # Create temp folder that is deleted when the function returns
    # Read taxonomy using read_taxonomy_leaf_nodes from instructlab-sdg package
    # The above step also downloads the reference documents.
    # Move all the downloaded documents under the temp folder
    # Pass the list to process_docs_from_folder
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Temporary directory created: {temp_dir}")
        knowledge_files = lookup_knowledge_files(taxonomy_path, taxonomy_base, temp_dir)
        logger.info(f"Found {len(knowledge_files)} knowledge files")
        logger.info(f"{knowledge_files}")

        process_docs_from_folder(temp_dir, output_dir)


def process_docs_from_folder(input_dir, output_dir):
    """
    Process user documents from a given `input_dir` folder to the given `output_dir` folder, using docling converters.
    Latest version of docling schema is used (currently, v2).
    """
    logger.info(f"Processing {input_dir} to {output_dir}")

    clear_directory(Path(output_dir))

    source_files = _load_source_files(input_dir=input_dir)
    logger.info(f"Transforming source files {[p.name for p in source_files]}")

    doc_converter = DocumentConverter()

    start_time = time.time()
    conv_results = doc_converter.convert_all(
        source_files,
        raises_on_error=False,
    )
    _, _, failure_count = _export_documents(conv_results, output_dir=Path(output_dir))
    end_time = time.time() - start_time
    logger.info(f"Document conversion complete in {end_time:.2f} seconds.")

    if failure_count > 0:
        raise RuntimeError(
            f"The example failed converting {failure_count} on {len(source_files)}."
        )


def _load_source_files(input_dir) -> list[Path]:
    return [
        Path(os.path.join(input_dir, f))
        for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f))
    ]


def _export_documents(
    conv_results: Iterable[ConversionResult],
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    failure_count = 0
    partial_success_count = 0

    for conv_res in conv_results:
        if conv_res.status == ConversionStatus.SUCCESS:
            success_count += 1
            doc_filename = conv_res.input.file.stem

            with (output_dir / f"{doc_filename}.json").open("w") as fp:
                fp.write(json.dumps(conv_res.document.export_to_dict()))
        elif conv_res.status == ConversionStatus.PARTIAL_SUCCESS:
            logger.info(
                f"Document {conv_res.input.file} was partially converted with the following errors:"
            )
            for item in conv_res.errors:
                print(f"\t{item.error_message}")
            partial_success_count += 1
        else:
            logger.info(f"Document {conv_res.input.file} failed to convert.")
            failure_count += 1

    logger.info(
        f"Processed {success_count + partial_success_count + failure_count} docs, "
        f"of which {failure_count} failed "
        f"and {partial_success_count} were partially converted."
    )
    return success_count, partial_success_count, failure_count
