# SPDX-License-Identifier: Apache-2.0

# Standard
import logging

# Third Party
import click

# First Party
from instructlab import configuration as config
from instructlab import log, utils
from instructlab.model.backends.llama import ServerException, server

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
@click.pass_context
@utils.display_params
def serve(
    ctx, model_path, gpu_layers, num_threads, max_ctx_size, model_family, log_file
):
    """Start a local server"""

    # Redirect server stdout and stderr to the logger
    log.stdout_stderr_to_logger(logger, log_file)

    logger.info(
        f"Using model '{model_path}' with {gpu_layers} gpu-layers and {max_ctx_size} max context size."
    )

    # First Party
    from instructlab.model.backends import llama

    host = ctx.obj.config.serve.host_port.split(":")[0]
    port = int(ctx.obj.config.serve.host_port.split(":")[1])

    # Instantiate the llama server
    llama_server = llama.Server(
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
