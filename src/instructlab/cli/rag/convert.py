# SPDX-License-Identifier: Apache-2.0

# Code to instantiate the ilab rag convert command.
# Calls out to code in ../../rag/convert.py to
# provide the implementation.

# Standard
import logging
import os

# Third Party
import click

# First Party
from instructlab import clickext
from instructlab.rag.convert import (
    convert_documents_from_folder,
    convert_documents_from_taxonomy,
)

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--input-dir",
    required=False,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=None,
    help="A folder with user documents to process. If it set, the taxonomy is ignored.  If it's missing, the knowledge taxonomy files will be processed instead.",
)
@click.option(
    "--taxonomy-path",
    required=True,
    type=click.Path(file_okay=False, readable=True),
    config_class="generate",
    cls=clickext.ConfigOption,
)
@click.option(
    "--taxonomy-base",
    required=True,
    config_class="generate",
    cls=clickext.ConfigOption,
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, writable=True),
    required=True,
    config_class="rag",
    config_sections="convert",
    cls=clickext.ConfigOption,
)
@click.pass_context
@clickext.display_params
def convert(
    ctx,  # pylint: disable=unused-argument
    taxonomy_path,
    taxonomy_base,
    input_dir,
    output_dir,
):
    """Pipeline to convert documents from their original format (e.g., PDF) into Docling JSON format for use by ilab rag ingest"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if input_dir is None:
        logger.info(
            f"Pre-processing latest taxonomy changes at {taxonomy_path}@{taxonomy_base}"
        )
        convert_documents_from_taxonomy(
            taxonomy_path=taxonomy_path,
            taxonomy_base=taxonomy_base,
            output_dir=output_dir,
        )
    else:
        logger.info(f"Pre-processing documents from {input_dir} to {output_dir}")
        convert_documents_from_folder(
            input_dir=input_dir,
            output_dir=output_dir,
        )
