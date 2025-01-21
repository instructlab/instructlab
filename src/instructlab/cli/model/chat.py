# SPDX-License-Identifier: Apache-2.0

# Standard
import logging
import pathlib

# Third Party
import click

# First Party
from instructlab import clickext
from instructlab import configuration as cfg
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
@click.pass_context
@clickext.display_params
def chat(
    ctx,
    question,
    model,
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
):
    """Runs a chat using the modified model"""
    chat_model(
        question,
        model,
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
        vllm_model_family=ctx.obj.config.serve.vllm.llm_family,
        vllm_args=ctx.obj.config.serve.vllm.vllm_args,
        max_startup_attempts=ctx.obj.config.serve.vllm.max_startup_attempts,
        logs_dir=ctx.obj.config.chat.logs_dir,
        vi_mode=ctx.obj.config.chat.vi_mode,
        visible_overflow=ctx.obj.config.chat.visible_overflow,
    )
