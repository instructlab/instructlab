# SPDX-License-Identifier: Apache-2.0

# Standard
import logging
import pathlib
import typing

# Third Party
import click

# First Party
from instructlab import clickext, log, utils
from instructlab.model.backends import backends
from instructlab.model.backends.backends import ServerException

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--model-path",
    cls=clickext.ConfigOption,
    type=click.Path(path_type=pathlib.Path),
    required=True,  # default from config
    help="Path to the model used during generation.",
)
@click.option(
    "--gpu-layers",
    cls=clickext.ConfigOption,
    config_section="llama_cpp",
    type=click.INT,
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
    cls=clickext.ConfigOption,
    config_section="llama_cpp",
    type=click.INT,
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
    "--backend",
    type=click.Choice(tuple(backends.SUPPORTED_BACKENDS)),
    cls=clickext.ConfigOption,
    required=False,  # auto-detect
    help=(
        "The backend to use for serving the model.\n"
        "Automatically detected based on the model file properties.\n"
    ),
)
@click.option(
    "--vllm-args",
    type=str,
    multiple=True,
    help=(
        "Specific arguments to pass into vllm. Each arg must be passed "
        "separately and surrounded by quotes.\n"
        " Example: --vllm-args='--dtype=auto' --vllm-args='--enable-lora'"
    ),
)
@click.pass_context
@utils.display_params
def serve(
    ctx: click.Context,
    model_path: pathlib.Path,
    gpu_layers: int,
    num_threads: int | None,
    max_ctx_size: int,
    model_family,
    log_file: pathlib.Path | None,
    backend: str | None,
    vllm_args: typing.Iterable[str],
) -> None:
    """Start a local server"""
    # First Party
    from instructlab.model.backends import llama_cpp, vllm

    host, port = utils.split_hostport(ctx.obj.config.serve.host_port)
    try:
        backend = backends.get(logger, model_path, backend)
    except (ValueError, AttributeError) as e:
        click.secho(f"Failed to determine backend: {e}", fg="red")
        raise click.exceptions.Exit(1)

    # Redirect server stdout and stderr to the logger
    log.stdout_stderr_to_logger(logger, log_file)

    logger.info(
        f"Using model '{model_path}' with {gpu_layers} gpu-layers and {max_ctx_size} max context size."
    )

    logger.info(f"Serving model '{model_path}' with {backend}")

    backend_instance: backends.BackendServer
    if backend == backends.LLAMA_CPP:
        # Instantiate the llama server
        if gpu_layers is None:
            gpu_layers = ctx.obj.config.serve.llama_cpp.gpu_layers
        if max_ctx_size is None:
            max_ctx_size = ctx.obj.config.serve.llama_cpp.max_ctx_size

        backend_instance = llama_cpp.Server(
            logger=logger,
            api_base=ctx.obj.config.serve.api_base(),
            model_path=model_path,
            model_family=model_family,
            host=host,
            port=port,
            gpu_layers=gpu_layers,
            max_ctx_size=max_ctx_size,
            num_threads=num_threads,
        )
    elif backend == backends.VLLM:
        # Instantiate the vllm server
        backend_instance = vllm.Server(
            logger=logger,
            api_base=ctx.obj.config.serve.api_base(),
            model_path=model_path,
            vllm_args=vllm_args,
            host=host,
            port=port,
        )
    else:
        click.secho(f"Unknown backend: {backend}", fg="red")
        raise click.exceptions.Exit(1)

    try:
        # Run the llama server
        backend_instance.run()

    except ServerException as exc:
        click.secho(f"Error creating server: {exc}", fg="red")
        raise click.exceptions.Exit(1)
