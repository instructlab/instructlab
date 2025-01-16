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
from instructlab.configuration import DEFAULTS
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
    help="The folder with user documents to process. In case it's missing, the knowledge taxonomy files will be processed instead.",
)
@click.option(
    "--taxonomy-path", required=False, type=click.Path(file_okay=False, readable=True)
)
@click.option("--taxonomy-base", required=False, cls=clickext.ConfigOption)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, writable=True),
    required=False,
    cls=clickext.ConfigOption,
)
@click.pass_context
@clickext.display_params
def convert(
    ctx,
    taxonomy_path,
    taxonomy_base,
    input_dir,
    output_dir,
):
    """Pipeline to convert documents from their original format (e.g., PDF) into Docling JSON format for use by ilab rag ingest"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if input_dir is None:
        if taxonomy_path is None:
            taxonomy_path = ctx.obj.config.generate.taxonomy_path
            if taxonomy_path is None:
                taxonomy_path = DEFAULTS.TAXONOMY_DIR
        if taxonomy_base is None:
            taxonomy_base = ctx.obj.config.generate.taxonomy_base
            if taxonomy_base is None:
                taxonomy_base = DEFAULTS.TAXONOMY_BASE

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
