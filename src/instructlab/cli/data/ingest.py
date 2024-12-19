# SPDX-License-Identifier: Apache-2.0

# Standard
import logging

# Third Party
import click

# First Party
from instructlab import clickext
from instructlab.data.ingest_docs import ingest_docs
from instructlab.data.taxonomy_utils import lookup_processed_documents_folder
from instructlab.defaults import DEFAULTS
from instructlab.rag.rag_configuration import (  # type: ignore
    document_store_configuration,
    embedder_configuration,
)

logger = logging.getLogger(__name__)


# TODO: fill-in help fields
@click.command()
@click.option(
    "--document-store-type",
    default="milvuslite",
    envvar="ILAB_DOCUMENT_STORE_TYPE",
    type=click.STRING,
    help="The document store type, one of: `milvuslite`.",
)
@click.option(
    "--document-store-uri",
    default="embeddings.db",
    envvar="ILAB_DOCUMENT_STORE_URI",
    type=click.STRING,
    help="The document store URI",
)
@click.option(
    "--document-store-collection-name",
    default="IlabEmbeddings",
    envvar="ILAB_DOCUMENT_STORE_COLLECTION_NAME",
    type=click.STRING,
    help="The document store collection name",
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
@click.option(
    "--output-dir",
    envvar="ILAB_OUTPUT_DIR",
    help="Directory where generated datasets are stored.",
)
@click.argument(
    "input_dir",
    required=False,
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
    document_store_type,
    document_store_uri,
    document_store_collection_name,
    model_dir,
    embedding_model_name,
    output_dir,
    input_dir,
):
    """The embedding ingestion pipeline"""

    document_store_config = document_store_configuration(
        type=document_store_type,
        uri=document_store_uri,
        collection_name=document_store_collection_name,
    )
    embedder_config = embedder_configuration(
        model_dir=model_dir,
        model_name=embedding_model_name,
    )
    logger.info(f"VectorDB params: {vars(document_store_config)}")
    logger.info(f"Embedding model: {vars(embedder_config)}")
    if not embedder_config.validate_local_model_path():
        raise click.UsageError(
            f"Cannot find local embedding model {embedding_model_name} in {model_dir}. Download the model before running the pipeline."
        )

    if input_dir is None:
        if output_dir is None:
            output_dir = ctx.obj.config.generate.output_dir
            if output_dir is None:
                output_dir = DEFAULTS.DATASETS_DIR
        logger.info(f"Ingesting latest taxonomy changes at {output_dir}")
        processed_docs_folder = lookup_processed_documents_folder(output_dir)
        if processed_docs_folder is None:
            click.secho(
                f"Cannot find the latest processed documents folders from {output_dir}."
                + " Please verify that you executed `ilab data generate` and you have updated or new knowledge"
                + " documents in the current taxonomy."
            )
            raise click.exceptions.Exit(1)

        logger.info(f"Latest processed docs are in {processed_docs_folder}")
        input_dir = processed_docs_folder

    ingest_docs(
        input_dir=input_dir,
        document_store_config=document_store_config,
        embedder_config=embedder_config,
    )

    return
