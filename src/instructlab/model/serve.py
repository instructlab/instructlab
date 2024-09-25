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
from instructlab.model.backends.common import ServerException
from instructlab.model.backends.server import BackendServer

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
# TODO: fix me, cli has model-family, but config option has llm-family :-/
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
    gpus: int | None,
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
    log.stdout_stderr_to_logger(
        logger=logger, log_file=log_file, fmt=ctx.obj.config.general.log_format
    )

    if chat_template is None:
        chat_template = ctx.obj.config.serve.chat_template

    logger.info(
        f"Using model '{model_path}' with {gpu_layers} gpu-layers and {max_ctx_size} max context size."
    )

    logger.info(f"Serving model '{model_path}' with {backend}")

    backend_instance: BackendServer
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
        # First Party
        from instructlab.model.backends.vllm import contains_argument

        # Warn if unsupported backend parameters are passed
        warn_for_unsupported_backend_param(ctx)

        ctx.obj.config.serve.vllm.vllm_args = ctx.obj.config.serve.vllm.vllm_args or []
        if gpus:
            if contains_argument(
                "--tensor-parallel-size", ctx.obj.config.serve.vllm.vllm_args
            ):
                logger.info(
                    "'--gpus' flag used alongside '--tensor-parallel-size' in the vllm_args section of the config file. Using value of the --gpus flag."
                )
            # even if there are 2 duplicate flags in vLLM args, vLLM uses the second flag
            ctx.obj.config.serve.vllm.vllm_args.extend(
                ["--tensor-parallel-size", str(gpus)]
            )

        # serve.vllm.vllm_args defaults to []
        vllm_args = ctx.obj.config.serve.vllm.vllm_args

        # Instantiate the vllm server
        if ctx.args:
            # any vllm flag included in ctx.args (click arguments after "--"),
            # has precedence over the value over the flags in serve.vllm.vllm_args
            # section of the config and the value of the flag `--gpus`.
            if gpus and contains_argument("--tensor-parallel-size", ctx.args):
                logger.info(
                    "'--gpus' flag used alongside '--tensor-parallel-size' flag in `ilab model serve`. Using value of the --tensor-parallel-size flag."
                )
            vllm_args.extend(ctx.args)

        backend_instance = vllm.Server(
            api_base=ctx.obj.config.serve.api_base(),
            model_family=model_family,
            model_path=model_path,
            chat_template=chat_template,
            vllm_args=vllm_args,
            host=host,
            port=port,
            vllm_path=ctx.obj.config.serve.vllm.vllm_path,
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


def warn_for_unsupported_backend_param(
    ctx: click.Context,
):
    for param in ["gpu_layers", "num_threads", "max_ctx_size"]:
        if ctx.get_parameter_source(param) == click.core.ParameterSource.COMMANDLINE:
            logger.warning(
                f"Option '--{param.replace('_','-')}' not supported by the backend."
            )
