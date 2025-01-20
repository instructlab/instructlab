# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import List, Tuple
import logging
import pathlib
import sys

# Third Party
import click

# First Party
from instructlab import log
from instructlab.configuration import write_config
from instructlab.model.backends import backends
from instructlab.model.backends.common import ServerException
from instructlab.model.backends.server import BackendServer
from instructlab.utils import contains_argument

logger = logging.getLogger(__name__)


def signal_handler(num_signal, __):
    """
    Signal handler for termination signals
    """
    print(f"Received termination signal {num_signal}, exiting...")
    sys.exit(0)


def get_gpu_count() -> int:
    # Third Party
    import torch

    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def check_gpu_count(gpus: int, available_gpus: int) -> bool:
    return gpus <= available_gpus


def check_tensor_parallel_size(
    vllm_args: List, available_gpus: int
) -> Tuple[bool, int]:
    """
    Checks if --tensor-parallel-size argument is valid.

    Parameters:
        vllm_args (List): Arguments from serve.vllm.vllm_args.
        available_gpus (int): Get it from torch.

    Returns:
        - (True, tensor_parallel_size) if --tensor-parallel-size is valid.
        - (False, tensor_parallel_size) if --tensor-parallel-size is invalid (greater than available GPUs) or not found.

    Example:
        For vllm_args = ["--tensor-parallel-size", "4"], if there are 2 GPUs:
            Returns: (False, 4)
    """
    # First Party
    from instructlab.model.backends.vllm import get_argument

    tensor_parallel_size = 0
    try:
        _, tps = get_argument("--tensor-parallel-size", vllm_args)
        if tps is not None:
            tensor_parallel_size = int(tps)
    except ValueError:
        return (False, -1)
    if tensor_parallel_size > available_gpus:
        return (
            False,
            tensor_parallel_size,
        )
    return (True, tensor_parallel_size)


def serve_backend(
    ctx,
    model_path: pathlib.Path,
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
    """Core server functionality to be called from the CLI"""
    # Configure logging
    root_logger = logging.getLogger()
    if log_file:
        log.add_file_handler_to_logger(root_logger, log_file)

    # First Party
    from instructlab.model.backends import llama_cpp, vllm

    try:
        backend = backends.get(model_path, backend)
    except (ValueError, AttributeError) as e:
        raise ValueError(f"Failed to determine backend: {e}") from e

    logger.info(f"Setting backend_type in the serve config to {backend}")
    ctx.obj.config.serve.server.backend_type = backend
    write_config(ctx.obj.config)
    if chat_template is None:
        chat_template = ctx.obj.config.serve.chat_template

    logger.info(
        f"Using model '{model_path}' with {gpu_layers} gpu-layers and {max_ctx_size} max context size."
    )
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
            log_file=log_file,
        )
    elif backend == backends.VLLM:
        # First Party
        from instructlab.cli.model.serve import warn_for_unsupported_backend_param

        warn_for_unsupported_backend_param(ctx)

        ctx.obj.config.serve.vllm.vllm_args = ctx.obj.config.serve.vllm.vllm_args or []

        available_gpus = get_gpu_count()

        if gpus:
            gpus_valid = check_gpu_count(gpus, available_gpus)
            if not gpus_valid:
                click.secho(
                    f"Specified --gpus value ({gpus}) exceeds available GPUs ({available_gpus}).\nPlease specify a valid number of GPUs.",
                    fg="red",
                )
                raise click.exceptions.Exit(1)
            if contains_argument(
                "--tensor-parallel-size", ctx.obj.config.serve.vllm.vllm_args
            ):
                logger.info(
                    "'--gpus' flag used alongside '--tensor-parallel-size' in the vllm_args section of the config file. Using value of the --gpus flag."
                )
            ctx.obj.config.serve.vllm.vllm_args.extend(
                ["--tensor-parallel-size", str(gpus)]
            )
        else:
            tps_size_valid, tensor_parallel_size = check_tensor_parallel_size(
                ctx.obj.config.serve.vllm.vllm_args, available_gpus
            )
            if not tps_size_valid and tensor_parallel_size != -1:
                click.secho(
                    f"Invalid --tensor-parallel-size ({tensor_parallel_size}) value. It cannot be greater than the number of available GPUs ({available_gpus}).\nPlease reduce --tensor-parallel-size to a valid value or ensure that sufficient GPUs are available.",
                    fg="red",
                )
                raise click.exceptions.Exit(1)

        vllm_args = ctx.obj.config.serve.vllm.vllm_args
        if ctx.args:
            vllm_args.extend(ctx.args)

        backend_instance = vllm.Server(
            api_base=ctx.obj.config.serve.api_base(),
            model_family=model_family,
            model_path=model_path,
            chat_template=chat_template,
            vllm_args=vllm_args,
            host=host,
            port=port,
            log_file=log_file,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

    try:
        backend_instance.run()
    except ServerException as exc:
        raise RuntimeError(f"Error creating server: {exc}") from exc
    except KeyboardInterrupt:
        logger.info("Server terminated by keyboard")
    finally:
        backend_instance.shutdown()
        raise SystemExit(0)
