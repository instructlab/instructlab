# SPDX-License-Identifier: Apache-2.0

# Standard
import logging

# Third Party
import click

# First Party
from instructlab import configuration as config
from instructlab import log, utils
from instructlab.model.backends.llama_cpp import ServerException, server

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--model-path",
    type=click.Path(),
    default=config.DEFAULT_MODEL_PATH,
    show_default=True,
    help="Path to the model used during generation.",
)
@click.option(
    "--gpu-layers",
    type=click.INT,
    help="The number of layers to put on the GPU. The rest will be on the CPU. Defaults to -1 to move all to GPU.",
)
@click.option("--num-threads", type=click.INT, help="The number of CPU threads to use.")
@click.option(
    "--max-ctx-size",
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
    type=click.Path(),
    help="Log file path to write server logs to.",
)
@click.option(
    "--backend",
    type=click.STRING,  # purposely not using click.Choice to allow for auto-detection
    help=(
        "The backend to use for serving the model."
        "Automatically detected based on the model file properties."
        "Supported: 'llama-cpp'."
    ),
)
@click.pass_context
@utils.display_params
def serve(
    ctx,
    model_path,
    gpu_layers,
    num_threads,
    max_ctx_size,
    model_family,
    log_file,
    backend,
):
    """Start a local server"""

    # First Party
    from instructlab.model.backends import llama_cpp

    host = ctx.obj.config.serve.host_port.split(":")[0]
    port = int(ctx.obj.config.serve.host_port.split(":")[1])

    # First Party
    from instructlab.model.backends import backends

    try:
        backend = backends.get(logger, model_path, backend)
    except ValueError as e:
        click.secho(f"Failed to determine backend: {e}", fg="red")
        raise click.exceptions.Exit(1)

    # Redirect server stdout and stderr to the logger
    log.stdout_stderr_to_logger(logger, log_file)

    logger.info(
        f"Using model '{model_path}' with {gpu_layers} gpu-layers and {max_ctx_size} max context size."
    )

    logger.info(f"Serving model '{model_path}' with {backend}")

    if backend == backends.LLAMA_CPP:
        # Instantiate the llama server
        llama_server = llama_cpp.Server(
            logger=logger,
            model_path=model_path,
            gpu_layers=gpu_layers,
            max_ctx_size=max_ctx_size,
            num_threads=num_threads,
            model_family=model_family,
            host=host,
            port=port,
        )

        try:
            # Run the llama server
            llama_server.run()
        except ServerException as exc:
            click.secho(f"Error creating server: {exc}", fg="red")
            raise click.exceptions.Exit(1)
