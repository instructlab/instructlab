# SPDX-License-Identifier: Apache-2.0

# This is the implementation code for the ilab rag convert command, which converts documents from their
# native format (e.g., PDF) to Docling JSON for use by ilab rag ingest.  See also the code in cli/rag/convert.py
# which instantiates the CLI command and calls out to the methods in this file.

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
from docling.datamodel.base_models import InputFormat  # type: ignore
from docling.datamodel.document import ConversionResult  # type: ignore
from docling.datamodel.pipeline_options import EasyOcrOptions  # type: ignore
from docling.datamodel.pipeline_options import OcrOptions  # type: ignore
from docling.datamodel.pipeline_options import PdfPipelineOptions  # type: ignore
from docling.datamodel.pipeline_options import TesseractOcrOptions  # type: ignore
from xdg_base_dirs import xdg_data_dirs, xdg_data_home
import yaml

# First Party
from instructlab.rag.taxonomy_utils import lookup_knowledge_files
from instructlab.utils import clear_directory

logger = logging.getLogger(__name__)


def convert_documents_from_taxonomy(taxonomy_path, taxonomy_base, output_dir):
    """
    Converts documents from a taxonomy. It uses the tempfile module to create a temporary directory that is
    deleted when the function returns. It then uses the lookup_knowledge_files function from the instructlab-sdg
    package to read the taxonomy and download the reference documents. The downloaded documents are moved to the
    temporary directory. Finally, the convert_documents_from_folder function is called to process the documents
    in the temporary directory and save them to the output directory.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        logger.info(f"Temporary directory created: {temp_dir}")
        knowledge_files = lookup_knowledge_files(taxonomy_path, taxonomy_base, temp_dir)
        logger.info(f"Found {len(knowledge_files)} knowledge files")
        logger.info(f"{knowledge_files}")

        convert_documents_from_folder(temp_dir, output_dir)


def convert_documents_from_folder(input_dir, output_dir):
    """
    Convert user documents from a given `input_dir` folder to the given `output_dir` folder, using docling converters.
    Latest version of docling schema is used (currently, v2).
    """
    logger.info(f"Processing {input_dir} to {output_dir}")

    clear_directory(Path(output_dir))

    source_files = _load_source_files(input_dir=input_dir)
    logger.info(f"Transforming source files {[p.name for p in source_files]}")

    doc_converter = _initialize_docling()
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
    """
    Takes an input directory as an argument and returns a list of paths to all the files in that directory.
    """
    return [
        Path(os.path.join(input_dir, f))
        for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f))
    ]


# From the Docling batch conversion example: https://ds4sd.github.io/docling/examples/batch_convert/
def _export_documents(
    conv_results: Iterable[ConversionResult],
    output_dir: Path,
):
    """
    Exports documents based on the conversion results.
    The function first creates the output directory if it does not exist. It then initializes three
    counters to keep track of the number of successful, partially successful, and failed conversions.
    Next, it iterates over each conversion. If the conversion status is ConversionStatus.SUCCESS, it
    increments the success_count and writes the document to a JSON file in the output directory.
    The document is exported to a dictionary using the export_to_dict() method and written to a
    file using the json.dumps() function.  If the conversion status is ConversionStatus.PARTIAL_SUCCESS,
    it increments the partial_success_count and logs a message indicating that the document was partially
    converted with the errors.  If the conversion status is ConversionStatus.FAILURE, it increments the
    failure_count and logs a message indicating that the document failed to convert.
    Finally, the function logs a message indicating the total number of documents processed, the number
    of successful conversions, the number of partially successful conversions, and
    the number of failed conversions. It then returns the three counters as a tuple.
    """
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


# Adapted from part of sdg/generate_data.py:_sdg_init and sdg/utils/chunkers.py:DocumentChunker
# because that code is being refactored so we want to avoid importing anything from it.
# TODO: Once the code base has settled down, we should make sure this code exists only in one place.
# TODO: Also, it is very messy to be pulling the docling_model_path from
#  {xdg_data_home()}/instructlab/sdg/config.yaml.  This seems like a non-standard way to configure
#  InstructLab functionality.  As part of unifying the code base, we should reconsider how this
#  is configured.
def _initialize_docling():
    """
    Initializes a Docling document converter for converting files (e.g., PDF) to Docling JSON format for use by
    the Docling chunkers.
    """
    data_dirs = [os.path.join(xdg_data_home(), "instructlab", "sdg")]
    data_dirs.extend(os.path.join(dir, "instructlab", "sdg") for dir in xdg_data_dirs())

    docling_model_path = None
    sdg_models_path = docling_model_path
    for d in data_dirs:
        if os.path.exists(os.path.join(d, "models")):
            sdg_models_path = os.path.join(d, "models")
            break

    if sdg_models_path is not None:
        try:
            with open(
                os.path.join(sdg_models_path, "config.yaml"), "r", encoding="utf-8"
            ) as file:
                config = yaml.safe_load(file)
                docling_model_path = config["models"][0]["path"]
        except (FileNotFoundError, NotADirectoryError, PermissionError) as e:
            logger.warning(f"unable to read docling models path from config.yaml {e}")

    pipeline_options = PdfPipelineOptions(
        artifacts_path=docling_model_path,
        do_ocr=False,
    )
    ocr_options = _resolve_ocr_options()
    if ocr_options is not None:
        pipeline_options.do_ocr = True
        pipeline_options.ocr_options = ocr_options

    _load_converter_and_format_options()
    doc_converter = DocumentConverter(  # pylint: disable=E0602
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)  # pylint: disable=E0602
        }
    )

    return doc_converter


# Adapted from sdg.utils.chunkers because that code is being refactored so we want to avoid importing anything from it.
# TODO: Once the code base has settled down, we should make sure this code exists only in one place.
def _resolve_ocr_options() -> OcrOptions:
    """
    Attempts to resolve OCR options for a PDF document. It first tries to use the Tesseract OCR library,
    and if that fails, it tries the EasyOCR library. If neither of these libraries are available, it
    returns None to indicate that OCR will not be used. Note that it imports the OCR libraries inside
    the code if/when they are needed because they are kind of heavy.
    """
    # First, attempt to use tesserocr
    try:
        ocr_options = TesseractOcrOptions()

        # pylint: disable=import-outside-toplevel
        # Third Party
        from docling.models.tesseract_ocr_model import (  # type: ignore[import-untyped]
            TesseractOcrModel,
        )

        _ = TesseractOcrModel(True, ocr_options)
        return ocr_options
    except ImportError:
        # No tesserocr, so try something else
        pass
    try:
        # pylint: disable=import-outside-toplevel
        # Third Party
        from docling.models.easyocr_model import (  # type: ignore[import-untyped]
            EasyOcrModel,
        )

        ocr_options = EasyOcrOptions()

        # Keep easyocr models on the CPU instead of GPU
        ocr_options.use_gpu = False
        # triggers torch loading, import lazily

        _ = EasyOcrModel(True, ocr_options)
        return ocr_options
    except ImportError:
        # no easyocr either, so don't use any OCR
        logger.error(
            "Failed to load Tesseract and EasyOCR - disabling optical character recognition in PDF documents"
        )
        return None


def _load_converter_and_format_options():
    """
    Used by the unit tests to force the load of these classes right before patching.  Also used by _initialize_docling.

    The main motivation here is that we don't want heavy dependencies like these to load if they are not used because
    they take time and use memory.  Anyone running any ilab command _other_ than `convert` will not need these dependencies
    loaded but if we statically loaded them in the header of this file, they would get loaded anyway.

    We have a test for this test_lab.py:test_ilab_cli_imports, which calls tests/testdata/leanimports.py, which has
    an explicit list of expensive libraries that we don't want statically loaded. That list includes torch, which
    Docling requires.  So we need to dynamically load these classes here.

    Of course, we could just put the imports into the place in this file where we use them, but that would make it
    hard for unit tests to replace the imports with mocks.  So instead, we make the imports global and load them in
    this function and then the unit tests call this function before they then replace DocumentConverter with a
    mock document converter.  See tests/test_lab_rag_convert.py for details.
    """
    global DocumentConverter, PdfFormatOption  # pylint: disable=W0603
    # pylint: disable=import-outside-toplevel
    # Third Party
    from docling.document_converter import (  # type: ignore[import-untyped]
        DocumentConverter,
        PdfFormatOption,
    )
