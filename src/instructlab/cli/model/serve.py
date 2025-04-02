# SPDX-License-Identifier: Apache-2.0

# Standard
import logging
import pathlib
import signal

# Third Party
import click

# First Party
from instructlab import clickext
from instructlab.configuration import resolve_model_id, write_config
from instructlab.defaults import DEFAULTS
from instructlab.model.backends import backends
from instructlab.model.serve_backend import serve_backend, signal_handler

logger = logging.getLogger(__name__)

# Register the signal handler for SIGTERM
signal.signal(signal.SIGTERM, signal_handler)


def warn_for_unsupported_backend_param(ctx):
    for param in ["gpu_layers", "num_threads", "max_ctx_size"]:
        if ctx.get_parameter_source(param) == click.core.ParameterSource.COMMANDLINE:
            logger.warning(
                f"Option '--{param.replace('_','-')}' not supported by the backend."
            )


@click.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": False},
)
@click.option(
    "--model-path",
    type=click.Path(path_type=pathlib.Path),
    cls=clickext.ConfigOption,
    required=True,  # default from config
)
@click.option(
    "--model-id",
    help="ID of the model to use for chatting from the config models list.",
    default=None,
)
@click.option(
    "--gpu-layers",
    type=click.INT,
    cls=clickext.ConfigOption,
    config_sections="llama_cpp",
    required=True,  # default from config
)
@click.option(
    "--num-threads",
    type=click.INT,
    required=False,
    help="The number of CPU threads to use.",
)
@click.option(
    "--max-ctx-size",
    type=click.INT,
    cls=clickext.ConfigOption,
    config_sections="llama_cpp",
)
@click.option(
    "--model-family",
    type=str,
    help="Model family is used to specify which chat template to serve with",
)
@click.option(
    "--log-file",
    type=click.Path(path_type=pathlib.Path),
    required=False,
    help="Log file path to write server logs to.",
)
@click.option(
    "--chat-template",
    type=str,
    cls=clickext.ConfigOption,
)
@click.option(
    "--backend",
    type=click.Choice(tuple(backends.SUPPORTED_BACKENDS)),
    cls=clickext.ConfigOption,
    required=False,  # auto-detect
)
@click.option(
    "--gpus",
    type=click.IntRange(min=0),
    cls=clickext.ConfigOption,
    config_sections="vllm",
)
@click.option(
    "-h",
    "--host",
    type=str,
    cls=clickext.ConfigOption,
    config_sections="server",
)
@click.option(
    "-p",
    "--port",
    type=int,
    cls=clickext.ConfigOption,
    config_sections="server",
)
@click.pass_context
@clickext.display_params
def serve(
    ctx: click.Context,
    model_path: pathlib.Path,
    model_id: str | None,
    gpu_layers: int,
    num_threads: int | None,
    max_ctx_size: int,
    model_family,
    log_file: pathlib.Path | None,
    backend: str | None,
    chat_template: str | None,
    gpus: int | None,
    host: str,
    port: int,
) -> None:
    """Starts a local server"""

    warn_for_unsupported_backend_param(ctx)

    # we need to keep track of the max_ctx_size being uses in the active server so we don't overflow the context
    if max_ctx_size != DEFAULTS.MAX_CONTEXT_SIZE:
        logger.info(
            f"Setting current_max_ctx_size in the serve config to {max_ctx_size}"
        )
        ctx.obj.config.serve.server.current_max_ctx_size = max_ctx_size
        write_config(ctx.obj.config)

    if model_id:
        try:
            model_config = resolve_model_id(model_id, ctx.obj.config.models)
            if not model_config:
                raise ValueError(
                    f"Model with ID '{model_id}' not found in the configuration."
                )
            model_path = pathlib.Path(model_config.path)
            model_family = model_config.family if model_config.family else model_family
        except ValueError as ve:
            click.secho(f"failed to locate model by ID: {ve}", fg="red")
            raise click.exceptions.Exit(1)

    serve_backend(
        ctx,
        model_path,
        gpu_layers,
        num_threads,
        max_ctx_size,
        model_family,
        log_file,
        backend,
        chat_template,
        gpus,
        host,
        port,
    )


if __name__ == "__main__":
    serve()
