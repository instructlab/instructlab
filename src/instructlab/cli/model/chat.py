# SPDX-License-Identifier: Apache-2.0

# Standard
import logging
import os
import pathlib

# Third Party
import click

# First Party
from instructlab import clickext
from instructlab import client_utils as ilabclient
from instructlab import configuration as cfg
from instructlab import log
from instructlab.client_utils import HttpClientParams

# Local
from ...client_utils import http_client

logger = logging.getLogger(__name__)


def is_openai_server_and_serving_model(
    endpoint: str, api_key: str, http_params: HttpClientParams
) -> bool:
    """
    Given an endpoint, returns whether or not the server is OpenAI-compatible
    and is actively serving at least one model.
    """
    try:
        models = ilabclient.list_models(
            endpoint, api_key=api_key, http_client=http_client(http_params)
        )
        return len(models.data) > 0
    except ilabclient.ClientException:
        return False


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
    # pylint: disable=import-outside-toplevel
    # First Party
    from instructlab.model.backends.common import is_temp_server_running
    from instructlab.model.chat import ChatException, chat_cli

    users_endpoint_url = cfg.get_api_base(ctx.obj.config.serve.host_port)

    # we prefer the given endpoint when one is provided, else we check if the user
    # is actively serving something before falling back to serving our own model
    backend_instance = None
    if endpoint_url:
        api_base = endpoint_url
    elif is_openai_server_and_serving_model(
        users_endpoint_url,
        api_key,
        http_params={
            "tls_client_cert": tls_client_cert,
            "tls_client_key": tls_client_key,
            "tls_client_passwd": tls_client_passwd,
            "tls_insecure": tls_insecure,
        },
    ):
        api_base = users_endpoint_url
        if serving_log_file:
            logger.warning(
                "Setting serving log file (--serving-log-file) is not supported when the server is already running"
            )
    else:
        # First Party
        from instructlab.model.backends import backends

        # If a log file is specified, write logs to the file
        root_logger = logging.getLogger()
        if serving_log_file:
            log.add_file_handler_to_logger(root_logger, serving_log_file)

        ctx.obj.config.serve.llama_cpp.llm_family = model_family
        backend_instance = backends.select_backend(
            ctx.obj.config.serve,
            model_path=model,
            log_file=serving_log_file,
        )
        try:
            # Run the llama server
            api_base = backend_instance.run_detached(http_client(ctx.params))
        except Exception as exc:
            click.secho(f"Failed to start server: {exc}", fg="red")
            raise click.exceptions.Exit(1)

    # if only the chat is running (`ilab model chat`) and the temp server is not, the chat interacts
    # in server mode (`ilab model serve` is running somewhere, or we are talking to another
    # OpenAI compatible endpoint).
    if not is_temp_server_running():
        # Try to get the model name right if we know we're talking to a local `ilab model serve`.
        #
        # If the model from the CLI and the one in the config are the same, use the one from the
        # server if they are different else let's use what the user provided
        #
        # 'model' will always get a value and never be None so it's hard to distinguish whether
        # the value came from the user input or the default value.
        # We can only assume that if the value is the same as the default value and the value
        # from the config is the same as the default value, then the user didn't provide a value
        # we then compare it with the value from the server to see if it's different
        if (
            # We need to get the base name of the model because the model path is a full path and
            # the once from the config is just the model name
            os.path.basename(model) == cfg.DEFAULTS.GRANITE_GGUF_MODEL_NAME
            and os.path.basename(ctx.obj.config.chat.model)
            == cfg.DEFAULTS.GRANITE_GGUF_MODEL_NAME
            and api_base == ctx.obj.config.serve.api_base()
        ):
            logger.debug(
                "No model was provided by the user as a CLI argument or in the config, will use the model from the server"
            )
            try:
                models = ilabclient.list_models(
                    api_base=api_base,
                    http_client=http_client(ctx.params),
                )

                # Currently, we only present a single model so we can safely assume that the first model
                server_model = models.data[0].id if models is not None else None

                # override 'model' with the first returned model if not provided so that the chat print
                # the model used by the server
                model = (
                    server_model
                    if server_model is not None
                    and server_model != ctx.obj.config.chat.model
                    else model
                )
                logger.debug(f"Using model from server {model}")
            except ilabclient.ClientException as exc:
                click.secho(
                    f"Failed to list models from {api_base}. Please check the API key and endpoint.",
                    fg="red",
                )
                # Right now is_temp_server() does not check if a subprocessed vllm is up
                # shut it down just in case an exception is raised in the try
                # TODO: revise is_temp_server to check if a vllm server is running
                if backend_instance is not None:
                    backend_instance.shutdown()
                raise click.exceptions.Exit(1) from exc

    try:
        chat_cli(
            ctx,
            api_base=api_base,
            config=ctx.obj.config.chat,
            question=question,
            model=model,
            context=context,
            session=session,
            qq=quick_question,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except ChatException as exc:
        click.secho(f"Executing chat failed with: {exc}", fg="red")
        raise click.exceptions.Exit(1)
    finally:
        if backend_instance is not None:
            backend_instance.shutdown()
