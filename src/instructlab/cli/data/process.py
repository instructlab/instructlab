# SPDX-License-Identifier: Apache-2.0

# Standard
import logging

# Third Party
import click

# First Party
from instructlab import clickext
from instructlab.data.ingest_docs import EmbeddingModel  # type: ignore
from instructlab.data.process_docs import process_docs  # type: ignore
from instructlab.defaults import DEFAULTS

logger = logging.getLogger(__name__)


# TODO: fill-in help fields
@click.command()
@click.option(
    "--input",
    "input_dir",
    required=True,
    default=None,
    envvar="ILAB_PROCESS_INPUT",
    help="The folder with user documents to process.",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
)
@click.option(
    "--output",
    "output_dir",
    required=True,
    default=None,
    envvar="ILAB_PROCESS_OUTPUT",
    help="The folder where processed artifacts are stored.",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, readable=True),
)
@click.option(
    "--model-dir",
    default=lambda: DEFAULTS.MODELS_DIR,
    envvar="ILAB_MODEL_DIR",
    show_default="The default system model location store, located in the data directory.",
    help="Base directories where models are stored.",
)
@click.option(
    "--embedding-model",
    "embedding_model_name",
    default="sentence-transformers/all-minilm-l6-v2",
    envvar="ILAB_EMBEDDING_MODEL_NAME",
    help="The embedding model name",
    type=click.STRING,
)
@click.pass_context
@clickext.display_params
def process(
    ctx,
    input_dir,
    output_dir,
    model_dir,
    embedding_model_name,
):
    """The document processing pipeline"""

    embedding_model = EmbeddingModel(
        model_dir=model_dir,
        model_name=embedding_model_name,
    )
    logger.info(f"Embedding model: {vars(embedding_model)}")
    if not embedding_model.validate_local_model_path():
        raise click.UsageError(
            f"Cannot find local embedding model {embedding_model_name} in {model_dir}. Download the model before running the pipeline."
        )

    logger.info(f"Pre-processing documents from {input_dir} to {output_dir}")
    tokenizer_model_path = embedding_model.local_model_path()
    process_docs(
        input_dir=input_dir,
        output_dir=output_dir,
        model_path=tokenizer_model_path,
    )
