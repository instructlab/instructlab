# SPDX-License-Identifier: Apache-2.0

# Standard
import logging
import pathlib

# Third Party
import click

# First Party
from instructlab import clickext
from instructlab import configuration as cfg
from instructlab.configuration import resolve_model_id
from instructlab.defaults import DEFAULTS
from instructlab.model.chat import chat_model

logger = logging.getLogger(__name__)


@click.command()
@click.argument(
    "question",
    nargs=-1,
    type=click.UNPROCESSED,
)
@click.option(
    "-m",
    "--model",
    cls=clickext.ConfigOption,
    required=True,  # default from config
)
@click.option(
    "--model-id",
    help="ID of the model to use for chatting from the config models list.",
    default=None,
)
@click.option(
    "-c",
    "--context",
    cls=clickext.ConfigOption,
    required=True,  # default from config
)
@click.option(
    "-s",
    "--session",
    type=click.File("r"),
    cls=clickext.ConfigOption,
)
@click.option(
    "-qq",
    "--quick-question",
    is_flag=True,
    help="Exit after answering question.",
)
@click.option(
    "--max-tokens",
    type=click.INT,
    cls=clickext.ConfigOption,
)
@click.option(
    "--endpoint-url",
    type=click.STRING,
    help="Custom URL endpoint for OpenAI-compatible API. Defaults to the `ilab model serve` endpoint.",
)
@click.option(
    "--api-key",
    type=click.STRING,
    default=cfg.DEFAULTS.API_KEY,  # Note: do not expose default API key
    help="API key for API endpoint. [default: config.DEFAULT_API_KEY]",
)
@click.option(
    "--tls-insecure",
    is_flag=True,
    help="Disable TLS verification.",
)
@click.option(
    "--tls-client-cert",
    type=click.Path(),
    default="",
    show_default=True,
    help="Path to the TLS client certificate to use.",
)
@click.option(
    "--tls-client-key",
    type=click.Path(),
    default="",
    show_default=True,
    help="Path to the TLS client key to use.",
)
@click.option(
    "--tls-client-passwd",
    type=click.STRING,
    default="",
    help="TLS client certificate password.",
)
@click.option(
    "--model-family",
    help="Force model family to use when picking a chat template",
)
@click.option(
    "--serving-log-file",
    type=click.Path(path_type=pathlib.Path),
    required=False,
    help="Log file path to write server logs to.",
)
@click.option(
    "-t",
    "--temperature",
    cls=clickext.ConfigOption,
)
@click.option(
    "--rag",
    "rag_enabled",
    default=False,
    is_flag=True,
    help="To enable the RAG pipeline.",
)
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
    "--retriever-embedding-model-path",
    "embedding_model_path",
    type=click.STRING,
    cls=clickext.ConfigOption,
    config_class="rag",
    config_sections="embedding_model",
)
@click.option(
    "--retriever-top-k",
    "top_k",
    type=click.INT,
    default=DEFAULTS.RETRIEVER_TOP_K,
    cls=clickext.ConfigOption,
    config_class="rag",
    config_sections="retriever",
)
@click.option(
    "-nd",
    "--no-decoration",
    is_flag=True,
    help="Disable decorations for chat responses.",
)
@click.option(
    "--system-prompt",
    type=click.STRING,
    cls=clickext.ConfigOption,
)
@click.pass_context
@clickext.display_params
def chat(
    ctx,
    question,
    model,
    model_id: str | None,
    context,
    session,
    quick_question,
    max_tokens,
    endpoint_url,
    api_key,
    tls_insecure,
    tls_client_cert,
    tls_client_key,
    tls_client_passwd,
    model_family,
    serving_log_file,
    temperature,
    rag_enabled,
    uri,
    collection_name,
    embedding_model_path,
    top_k,
    no_decoration,
    system_prompt,
):
    """Runs a chat using the modified model"""

    if model_id:
        try:
            model_config = resolve_model_id(model_id, ctx.obj.config.models)
            if not model_config:
                raise ValueError(
                    f"Model with ID '{model_id}' not found in the configuration."
                )
            model = model_config.path
            model_family = model_config.family if model_config.family else model_family
        except ValueError as ve:
            click.secho(f"failed to locate model by ID: {ve}", fg="red")
            raise click.exceptions.Exit(1)

    chat_model(
        question,
        model,
        model_id,
        context,
        session,
        quick_question,
        max_tokens,
        endpoint_url,
        api_key,
        tls_insecure,
        tls_client_cert,
        tls_client_key,
        tls_client_passwd,
        model_family,
        serving_log_file,
        temperature,
        rag_enabled,
        uri,
        collection_name,
        embedding_model_path,
        top_k,
        no_decoration,
        system_prompt,
        backend_type=ctx.obj.config.serve.server.backend_type,
        host=ctx.obj.config.serve.server.host,
        port=ctx.obj.config.serve.server.port,
        current_max_ctx_size=ctx.obj.config.serve.server.current_max_ctx_size,
        params=ctx.params,
        backend_name=ctx.obj.config.serve.backend,
        chat_template=ctx.obj.config.serve.chat_template,
        api_base=ctx.obj.config.serve.api_base(),
        gpu_layers=ctx.obj.config.serve.llama_cpp.gpu_layers,
        max_ctx_size=ctx.obj.config.serve.llama_cpp.max_ctx_size,
        vllm_model_family=model_family,  # use the resolved model family
        vllm_args=ctx.obj.config.serve.vllm.vllm_args,
        max_startup_attempts=ctx.obj.config.serve.vllm.max_startup_attempts,
        logs_dir=ctx.obj.config.chat.logs_dir,
        vi_mode=ctx.obj.config.chat.vi_mode,
        visible_overflow=ctx.obj.config.chat.visible_overflow,
        models_config=ctx.obj.config.models,
    )
