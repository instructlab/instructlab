# Standard
import logging

# Third Party
import click

# First Party
from instructlab import clickext
from instructlab.defaults import DEFAULTS
from instructlab.rag.taxonomy_utils import lookup_processed_documents_folder

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--document-store-uri",
    "uri",
    type=click.STRING,
    cls=clickext.ConfigOption,
    config_sections="document_store",
)
@click.option(
    "--document-store-collection-name",
    "collection_name",
    type=click.STRING,
    cls=clickext.ConfigOption,
    config_sections="document_store",
)
@click.option(
    "--embedding-model-name",
    type=click.STRING,
    cls=clickext.ConfigOption,
    config_sections="embedding_model",
)
@click.option(
    "--input",
    "input_dir",
    required=False,
    type=click.Path(
        exists=True,
        dir_okay=True,
        file_okay=False,
    ),
    help="Directory where pre-processed documents are located.",
)
@click.pass_context
@clickext.display_params
def ingest(
    ctx,
    uri,
    collection_name,
    embedding_model_name,
    input_dir,
):
    """The embedding ingestion pipeline"""
    logger.info(f"VectorDB params: {collection_name} @ {uri}")
    logger.info(f"Embedding model: {embedding_model_name}")

    if input_dir is None:
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

    # First Party
    from instructlab.rag.document_store_factory import create_document_store_ingestor

    ingestor = create_document_store_ingestor(
        document_store_uri=uri,
        document_store_collection_name=collection_name,
        embedding_model_path=embedding_model_name,
    )
    ingestor.ingest_documents(input_dir=input_dir)
