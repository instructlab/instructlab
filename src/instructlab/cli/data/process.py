# SPDX-License-Identifier: Apache-2.0

# Standard
import logging

# Third Party
import click

# First Party
from instructlab import clickext
from instructlab.data.process_docs import process_docs  # type: ignore

logger = logging.getLogger(__name__)


# TODO: fill-in help fields
@click.command()
@click.option(
    "--input",
    "input_dir",
    required=False,
    default=None,
    envvar="ILAB_PROCESS_INPUT",
    help="The folder with user documents to process.",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
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
    input_dir,
    output_dir,
):
    """The document processing pipeline"""

    logger.info(f"Pre-processing documents from {input_dir} to {output_dir}")
    process_docs(
        input_dir=input_dir,
        output_dir=output_dir,
    )
