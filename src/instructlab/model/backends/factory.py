# Standard
from typing import Optional
import datetime
import json
import logging
import os
import pathlib
import sys
import time

# Third Party
import click

# First Party
from instructlab import utils
from instructlab.model.backends import backends, llama_cpp, vllm


def backend_factory(
    ctx,
    logger,
    model_family,
    tls_insecure,
    tls_client_cert,
    tls_client_key,
    tls_client_passwd,
):
    model_path = pathlib.Path(ctx.obj.config.serve.model_path)
    backend = ctx.obj.config.serve.backend
    try:
        backend = backends.get(logger, model_path, backend)
    except ValueError as e:
        click.secho(f"Failed to determine backend: {e}", fg="red")
        raise click.exceptions.Exit(1)

    host, port = utils.split_hostport(ctx.obj.config.serve.host_port)

    if backend == backends.LLAMA_CPP:
        # Instantiate the llama server
        backend_instance = llama_cpp.Server(
            logger=logger,
            api_base=ctx.obj.config.serve.api_base(),
            model_path=model_path,
            gpu_layers=ctx.obj.config.serve.gpu_layers,
            max_ctx_size=ctx.obj.config.serve.max_ctx_size,
            num_threads=None,  # exists only as a flag not a config
            model_family=model_family,
            host=host,
            port=port,
        )

    if backend == backends.VLLM:
        # Instantiate the vllm server
        backend_instance = vllm.Server(
            logger=logger,
            api_base=ctx.obj.config.serve.api_base(),
            model_path=model_path,
            model_family=model_family,
            host=host,
            port=port,
        )

    try:
        # Run the server
        backend_instance.run_detached(
            tls_insecure, tls_client_cert, tls_client_key, tls_client_passwd
        )
        # api_base will be set by run_detached
    except Exception as exc:
        click.secho(f"Failed to start server: {exc}", fg="red")
        raise click.exceptions.Exit(1)
    return backend_instance
