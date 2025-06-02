# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import List
import logging
import pathlib
import sys

# First Party
from instructlab import log
from instructlab.config.init import is_hpu_available
from instructlab.configuration import write_config
from instructlab.model.backends import backends
from instructlab.model.backends.common import ServerException
from instructlab.model.backends.server import BackendServer

logger = logging.getLogger(__name__)


def signal_handler(num_signal, __):
    """
    Signal handler for termination signals
    """
    print(f"Received termination signal {num_signal}, exiting...")
    sys.exit(0)


def get_tensor_parallel_size(vllm_args: List) -> int:
    """
    Get the value of --tensor-parallel-size.

    Parameters:
        vllm_args (List): Arguments from serve.vllm.vllm_args.

    Returns:
        - Return tensor_parallel_size value.
    """
    # First Party
    from instructlab.model.backends.vllm import get_argument

    tensor_parallel_size = 0
    try:
        _, tps = get_argument("--tensor-parallel-size", vllm_args)
        if tps is not None:
            tensor_parallel_size = int(tps)
    except ValueError:
        logger.warning("Cannot find a valid value for --tensor-parallel-size.")
        return 0
    return tensor_parallel_size


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
        # Third Party
        import torch

        # First Party
        from instructlab.cli.model.serve import warn_for_unsupported_backend_param
        from instructlab.utils import contains_argument

        warn_for_unsupported_backend_param(ctx)

        ctx.obj.config.serve.vllm.vllm_args = ctx.obj.config.serve.vllm.vllm_args or []

        # TODO: add support for Gaudi, ROCm, etc.
        # we only offer HW detection if CUDA is available
        available_gpus = None
        no_rocm_or_hpu = torch.version.hip is None and not is_hpu_available()
        if torch.cuda.is_available() and no_rocm_or_hpu:
            available_gpus = torch.cuda.device_count()
        if gpus:
            if available_gpus and gpus > available_gpus:
                logger.error(
                    f"Specified --gpus value ({gpus}) exceeds available GPUs ({available_gpus}).\nPlease specify a valid number of GPUs."
                )
                raise SystemExit(1)
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
            tensor_parallel_size = get_tensor_parallel_size(
                ctx.obj.config.serve.vllm.vllm_args
            )
            if available_gpus and tensor_parallel_size > available_gpus:
                logger.error(
                    f"Invalid --tensor-parallel-size ({tensor_parallel_size}) value. It cannot be greater than the number of available GPUs ({available_gpus}).\nPlease reduce --tensor-parallel-size to a valid value or ensure that sufficient GPUs are available."
                )
                raise SystemExit(1)

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
