# SPDX-License-Identifier: Apache-2.0

# Standard
import logging

# Third Party
import click

# First Party
from instructlab import clickext
from instructlab.data.ingest_docs import (  # type: ignore
    EmbeddingModel,
    VectorDbParams,
    ingest_docs,
)
from instructlab.defaults import DEFAULTS

logger = logging.getLogger(__name__)


# TODO: fill-in help fields
@click.command()
@click.option(
    "--vectordb-type",
    default="milvuslite",
    envvar="ILAB_VECTORDB_TYPE",
    type=click.STRING,
    help="The vector DB type, one of: `milvuslite`, `milvus`.",
)
@click.option(
    "--vectordb-uri",
    default="embeddings.db",
    envvar="ILAB_VECTORDB_URI",
    type=click.STRING,
    help="The vector DB URI",
)
@click.option(
    "--vectordb-collection-name",
    default="Ilab",
    envvar="ILAB_VECTORDB_COLLECTION_NAME",
    type=click.STRING,
    help="The vector DB collection name",
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
    type=click.STRING,
    help="The embedding model name",
)
@click.argument(
    "input_dir",
    type=click.Path(
        exists=True,
        dir_okay=True,
        file_okay=False,
    ),
)
@click.pass_context
@clickext.display_params
def ingest(
    ctx,
    vectordb_type,
    vectordb_uri,
    vectordb_collection_name,
    model_dir,
    embedding_model_name,
    input_dir,
):
    """The embedding ingestion pipeline"""

    vectordb_params = VectorDbParams(
        vectordb_type=vectordb_type,
        vectordb_uri=vectordb_uri,
        vectordb_collection_name=vectordb_collection_name,
    )
    embedding_model = EmbeddingModel(
        model_dir=model_dir,
        model_name=embedding_model_name,
    )
    logger.info(f"VectorDB params: {vars(vectordb_params)}")
    logger.info(f"Embedding model: {vars(embedding_model)}")
    if not embedding_model.validate_local_model_path():
        raise click.UsageError(
            f"Cannot find local embedding model {embedding_model_name} in {model_dir}. Download the model before running the pipeline."
        )

    ingest_docs(
        input_dir=input_dir,
        vectordb_params=vectordb_params,
        embedding_model=embedding_model,
    )

    return
