# SPDX-License-Identifier: Apache-2.0

# Standard
import logging
import pathlib
import signal
import sys

# Third Party
import click

# First Party
from instructlab import clickext, log, utils
from instructlab.model.backends import backends
from instructlab.model.backends.backends import ServerException

logger = logging.getLogger(__name__)


def signal_handler(
    num_signal,
    __,
):
    """
    Signal handler for termination signals
    """
    print(f"Received termination signal {num_signal}, exiting...")
    sys.exit(0)


# Register the signal handler for SIGTERM
signal.signal(signal.SIGTERM, signal_handler)


@click.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": False},
)
@click.option(
    "--model-path",
    type=click.Path(path_type=pathlib.Path),
    cls=clickext.ConfigOption,
    required=True,  # default from config
    help="Path to the model used during generation.",
)
@click.option(
    "--gpu-layers",
    type=click.INT,
    cls=clickext.ConfigOption,
    config_sections="llama_cpp",
    required=True,  # default from config
    help="The number of layers to put on the GPU. -1 moves all layers. The rest will be on the CPU.",
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
    help="The context size is the maximum number of tokens considered by the model, for both the prompt and response. Defaults to 4096.",
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
    help=(
        "Which chat template to use in serving the model, either 'auto', "
        "'tokenizer', or a path to a jinja formatted template file. \n"
        " 'auto' (the default) indicates serve will decide which template to use.\n"
        " 'tokenizer' indicates the model's tokenizer config will be preferred"
    ),
)
@click.option(
    "--backend",
    type=click.Choice(tuple(backends.SUPPORTED_BACKENDS)),
    cls=clickext.ConfigOption,
    required=False,  # auto-detect
    help=(
        "The backend to use for serving the model.\n"
        "Automatically detected based on the model file properties.\n"
    ),
)
@click.pass_context
@clickext.display_params
def serve(
    ctx: click.Context,
    model_path: pathlib.Path,
    gpu_layers: int,
    num_threads: int | None,
    max_ctx_size: int,
    model_family,
    log_file: pathlib.Path | None,
    backend: str | None,
    chat_template: str | None,
) -> None:
    """Starts a local server

    The vLLM backend accepts additional parameters in the form of extra
    arguments after "--" separator:

      $ ilab model serve ... --backend=vllm -- --dtype=auto --enable-lora

    vLLm parameters are documented at
    https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html
    """
    # First Party
    from instructlab.model.backends import llama_cpp, vllm

    host, port = utils.split_hostport(ctx.obj.config.serve.host_port)
    try:
        backend = backends.get(model_path, backend)
    except (ValueError, AttributeError) as e:
        click.secho(f"Failed to determine backend: {e}", fg="red")
        raise click.exceptions.Exit(1)

    # Redirect server stdout and stderr to the logger
    log.stdout_stderr_to_logger(logger, log_file)

    if chat_template is None:
        chat_template = ctx.obj.config.serve.chat_template

    logger.info(
        f"Using model '{model_path}' with {gpu_layers} gpu-layers and {max_ctx_size} max context size."
    )

    logger.info(f"Serving model '{model_path}' with {backend}")

    backend_instance: backends.BackendServer
    if backend == backends.LLAMA_CPP:
        if ctx.args:
            ctx.fail(f"Unsupported extra arguments: {', '.join(ctx.args)}")
        backend_instance = llama_cpp.Server(
            api_base=ctx.obj.config.serve.api_base(),
            model_path=model_path,
            model_family=model_family,
            chat_template=chat_template,
            host=host,
            port=port,
            gpu_layers=gpu_layers,
            max_ctx_size=max_ctx_size,
            num_threads=num_threads,
        )
    elif backend == backends.VLLM:
        # Warn if unsupported backend parameters are passed
        warn_for_unsuported_backend_param(ctx)

        # Instantiate the vllm server
        if ctx.args:
            # extra click arguments after "--"
            vllm_args = ctx.args
        elif ctx.obj.config.serve.vllm.vllm_args:
            vllm_args = ctx.obj.config.serve.vllm.vllm_args
        else:
            vllm_args = []

        backend_instance = vllm.Server(
            api_base=ctx.obj.config.serve.api_base(),
            model_family=model_family,
            model_path=model_path,
            chat_template=chat_template,
            vllm_args=vllm_args,
            host=host,
            port=port,
        )
    else:
        click.secho(f"Unknown backend: {backend}", fg="red")
        raise click.exceptions.Exit(1)

    try:
        # Run the backend server
        backend_instance.run()

    except ServerException as exc:
        click.secho(f"Error creating server: {exc}", fg="red")
        raise click.exceptions.Exit(1)

    except KeyboardInterrupt:
        logger.info("Server terminated by keyboard")

    finally:
        backend_instance.shutdown()
        raise click.exceptions.Exit(0)


def warn_for_unsuported_backend_param(
    ctx: click.Context,
):
    for param in ["gpu_layers", "num_threads", "max_ctx_size"]:
        if ctx.get_parameter_source(param) == click.core.ParameterSource.COMMANDLINE:
            logger.warning(
                f"Option '--{param.replace('_','-')}' not supported by the backend."
            )
