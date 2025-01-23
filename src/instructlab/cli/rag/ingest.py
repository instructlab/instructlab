# Standard
import logging

# Third Party
import click

# First Party
from instructlab import clickext
from instructlab.defaults import DEFAULTS
from instructlab.feature_gates import FeatureGating, FeatureScopes, GatedFeatures

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--document-store-uri",
    "uri",
    type=click.STRING,
    cls=clickext.ConfigOption,
    config_class="rag",
    config_sections="document_store",
)
@click.option(
    "--document-store-collection-name",
    "collection_name",
    type=click.STRING,
    cls=clickext.ConfigOption,
    config_class="rag",
    config_sections="document_store",
)
@click.option(
    "--embedding-model-path",
    "embedding_model_path",
    type=click.STRING,
    cls=clickext.ConfigOption,
    config_class="rag",
    config_sections="embedding_model",
)
@click.option(
    "--input-dir",
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
    embedding_model_path,
    input_dir,
):
    """The embedding ingestion pipeline"""

    if not FeatureGating.feature_available(GatedFeatures.RAG):
        click.echo(
            f"This functionality is experimental; set {FeatureGating.env_var_name}"
            f' to "{FeatureScopes.DevPreviewNoUpgrade.value}" to enable.'
        )
        return

    logger.debug(f"Document Store: {collection_name} @ {uri}")
    logger.debug(f"Embedding model: {embedding_model_path}")

    if input_dir is None:
        # Local
        from ...rag.taxonomy_utils import lookup_processed_documents_folder

        output_dir = ctx.obj.config.generate.output_dir
        if output_dir is None:
            output_dir = DEFAULTS.DATASETS_DIR
            logger.warning(
                f"`output_dir` not defined in config. Using default path {output_dir}"
            )
        logger.info(f"Ingesting latest taxonomy changes at {output_dir}")
        processed_docs_folder = lookup_processed_documents_folder(output_dir)
        if processed_docs_folder is None:
            click.secho(
                f"Cannot find the latest processed documents folders from {output_dir}."
                + " Please verify that you executed `ilab data generate` and you have updated or new knowledge"
                + " documents in the current taxonomy.",
                fg="red",
            )
            raise click.exceptions.Exit(1)

        logger.info(f"Latest processed docs are in {processed_docs_folder}")
        input_dir = processed_docs_folder

    # First Party
    from instructlab.rag.document_store_factory import create_document_store_ingestor

    ingestor = create_document_store_ingestor(
        document_store_uri=uri,
        document_store_collection_name=collection_name,
        embedding_model_path=embedding_model_path,
    )
    ingestor.ingest_documents(input_dir=input_dir)
