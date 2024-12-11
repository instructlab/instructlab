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
from instructlab.data.transform_docs import transform_docs  # type: ignore
from instructlab.defaults import DEFAULTS

logger = logging.getLogger(__name__)


# TODO: fill-in help fields
@click.command()
@click.option(
    "--rag",
    "is_rag",
    is_flag=True,
    envvar="ILAB_RAG",
    help="Whether to process RAG artifacts.",
)
@click.option(
    "--transform",
    "is_transform",
    is_flag=True,
    envvar="ILAB_TRANSFORM",
)
@click.option(
    "--transform-output",
    default=None,
    envvar="ILAB_TRANSFORM_OUTPUT",
    type=click.Path(file_okay=False, dir_okay=True),
)
@click.option(
    "--vectordb-type",
    default="milvuslite",
    envvar="ILAB_VECTORDB_TYPE",
    type=click.STRING,
    help="The vector DB type, one of: `milvuslite`, `milvus`.",
)
@click.option(
    "--vectordb-uri",
    default="rag-output.db",
    envvar="ILAB_VECTORDB_URI",
    type=click.STRING,
    help="The vector DB URI",
)
@click.option(
    "--vectordb-collection-name",
    default="IlabEmbeddings",
    envvar="ILAB_VECTORDB_COLLECTION_NAME",
    type=click.STRING,
    help="The vector DB collection name",
)
@click.option(
    "--vectordb-authentication",
    default=None,
    envvar="ILAB_VECTORDB_AUTHENTICATION",
    type=click.STRING,
    hide_input=True,
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
)
@click.option(
    "--embedding-model-token",
    default=None,
    envvar="ILAB_EMBEDDING_MODEL_TOKEN",
    type=click.STRING,
    hide_input=True,
)
@click.argument(
    "path",
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
    is_rag,
    is_transform,
    transform_output,
    vectordb_type,
    vectordb_uri,
    vectordb_collection_name,
    vectordb_authentication,
    model_dir,
    embedding_model_name,
    embedding_model_token,
    path,
):
    """The embedding ingestion pipeline"""

    if is_rag:
        vectordb_params = VectorDbParams(
            vectordb_type=vectordb_type,
            vectordb_uri=vectordb_uri,
            vectordb_collection_name=vectordb_collection_name,
            vectordb_authentication=vectordb_authentication,
        )
        embedding_model = EmbeddingModel(
            model_dir=model_dir,
            model_name=embedding_model_name,
            model_token=embedding_model_token,
        )
        logger.info(f"VectorDB params: {vars(vectordb_params)}")
        logger.info(f"Embedding model: {vars(embedding_model)}")
        if not embedding_model.validate_local_model_path():
            raise click.UsageError(
                f"Cannot find local embedding model {embedding_model_name} in {model_dir}. Download the model before running the pipeline."
            )

        artifacts_dir = path
        if is_transform:
            logger.info(f"Pre-processing documents from {path} to {transform_output}")
            if not transform_output:
                raise click.UsageError(
                    "--transform-output must be provided when --transform is enabled."
                )
            tokenizer_model_path = embedding_model.local_model_path()
            transform_docs(
                input_dir=path,
                output_dir=transform_output,
                model_path=tokenizer_model_path,
            )
            artifacts_dir = transform_output

        ingest_docs(
            input_dir=artifacts_dir,
            vectordb_params=vectordb_params,
            embedding_model=embedding_model,
        )

        return

    click.secho(
        "RAG not enabled. Please use the --rag flag to execute the embedding ingestion pipeline.",
        fg="red",
    )
    raise click.exceptions.Exit(1)
