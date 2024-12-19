# SPDX-License-Identifier: Apache-2.0

# Standard
import logging

# Third Party
import click

# First Party
from instructlab import clickext
from instructlab.configuration import DEFAULTS
from instructlab.data.process_docs import (
    process_docs_from_folder,
    process_docs_from_taxonomy,
)

logger = logging.getLogger(__name__)


# TODO: fill-in help fields
@click.command()
@click.option(
    "--input",
    "input_dir",
    required=False,
    default=None,
    envvar="ILAB_PROCESS_INPUT",
    help="The folder with user documents to process. In case it's missing, the knowledge taxonomy files will be processed instead.",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
)
@click.option(
    "--taxonomy-path",
    type=click.Path(),
    envvar="ILAB_TAXONOMY_PATH",
    help="Directory where taxonomy is stored and accessed from.",
)
@click.option(
    "--taxonomy-base",
    envvar="ILAB_TAXONOMY_BASE",
    help="Branch of taxonomy used to calculate diff against.",
)
@click.argument(
    "output_dir",
    type=click.Path(
        exists=True,
        dir_okay=True,
        file_okay=False,
    ),
)
@click.pass_context
@clickext.display_params
def process(
    ctx,
    taxonomy_path,
    taxonomy_base,
    input_dir,
    output_dir,
):
    """The document processing pipeline"""

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
        process_docs_from_taxonomy(
            taxonomy_path=taxonomy_path,
            taxonomy_base=taxonomy_base,
            output_dir=output_dir,
        )
    else:
        logger.info(f"Pre-processing documents from {input_dir} to {output_dir}")
        process_docs_from_folder(
            input_dir=input_dir,
            output_dir=output_dir,
        )
