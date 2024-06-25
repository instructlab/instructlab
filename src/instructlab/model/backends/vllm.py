# SPDX-License-Identifier: Apache-2.0

# Standard
from contextlib import redirect_stderr, redirect_stdout
from time import sleep
from typing import Optional, Set
import asyncio
import logging
import multiprocessing
import os
import pathlib
import signal
import subprocess
import sys

# Third Party
from uvicorn import Config
import fastapi

# First Party
from instructlab import client

# Local
from ...client import ClientException, list_models
from ...configuration import get_api_base, get_model_family
from .backends import (
    API_ROOT_WELCOME_MESSAGE,
    BackendServer,
    ServerException,
    UvicornServer,
    ensure_server,
    get_uvicorn_config,
    is_temp_server_running,
    templates,
)


class Server(BackendServer):
    def __init__(
        self,
        logger: logging.Logger,
        api_base: str,
        model_path: pathlib.Path,
        model_family: str,
        host: str,
        port: int,
        process: multiprocessing.Process = None,
    ):
        super().__init__(logger, api_base, model_path, host, port)
        self.api_base = api_base
        self.model_path = model_path
        self.model_family = model_family
        self.process = process

    def run(self):
        """Start an OpenAI-compatible server with vllm"""
        try:
            server(
                self.model_path.as_posix(),
                self.model_family,
                self.host,
                self.port,
            )
        except ServerException as exc:
            raise exc

    def create_server_process(self, port: str) -> multiprocessing.Process:
        mpctx = multiprocessing.get_context(None)
        server_process = mpctx.Process(
            target=server,
            kwargs={
                "model_path": self.model_path.as_posix(),
                "model_family": self.model_family,
                "port": port,
                "host": self.host,
            },
        )

        return server_process

    def run_detached(
        self, tls_insecure, tls_client_cert, tls_client_key, tls_client_passwd
    ):
        try:
            server_process, api_base = ensure_server(
                logger=self.logger,
                server_process_func=self.create_server_process,
                api_base=self.api_base,
                host=self.host,
                port=self.port,
                tls_insecure=tls_insecure,
                tls_client_cert=tls_client_cert,
                tls_client_key=tls_client_key,
                tls_client_passwd=tls_client_passwd,
            )
            self.process = server_process
            self.api_base = api_base
        except ServerException as exc:
            raise exc

    def shutdown(self):
        """Shutdown vllm server"""
        # Needed when the chat is connected to a server and the temporary server is started
        if self.process is not None:
            self.logger.debug("Terminating temporary server.")
            self.process.terminate()
            self.process.join(timeout=30)

        else:
            self.logger.debug("No server process to terminate.")


# Heavily inspired by https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/api_server.py
def server(
    model_path,
    model_family,
    host="localhost",
    port=8000,
):
    # Third Party
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
    from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
    from vllm.usage.usage_lib import UsageContext
    import vllm.entrypoints.openai.api_server as api_server

    app = fastapi.FastAPI()

    engine_args = AsyncEngineArgs(
        model=model_path,
        max_model_len=4096,  # TODO: make this configurable
    )

    engine = api_server.engine = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.OPENAI_API_SERVER
    )
    engine.engine_use_ray = False

    model_config = asyncio.run(engine.get_model_config())

    template = ""
    # TODO: find the equivalent in vllm for eos_token and bos_token
    eos_token = "<|endoftext|>"
    bos_token = ""
    for template_dict in templates:
        if template_dict["model"] == get_model_family(model_family, model_path):
            template = template_dict["template"]
            if template_dict["model"] == "mixtral":
                eos_token = "</s>"
                bos_token = "<s>"

    api_server.openai_serving_chat = OpenAIServingChat(
        engine,
        model_config,
        served_model_names=[model_path],
        response_role="assistant",
        lora_modules=[],
        chat_template=template,  # chat_template can be a file too, should we support that?
    )
    api_server.openai_serving_completion = OpenAIServingCompletion(
        engine, model_config, served_model_names=[model_path], lora_modules=[]
    )

    @app.get("/")
    def read_root():
        return {"message": API_ROOT_WELCOME_MESSAGE}

    # Add the routes from the imported api_server app to your new app
    # NOTE(leseb): We cannot use api_server.app directly since it's using a lifespan that check for "engine_args"
    # and somehow I couldn't figure out why, here is the erorr:
    #
    # vllm/entrypoints/openai/api_server.py", line 52, in lifespan if not engine_args.disable_log_stats: NameError: name 'engine_args' is not defined
    for route in api_server.app.routes:
        app.router.routes.append(route)

    config = get_uvicorn_config(
        app=app,
        host=host,
        port=port,
    )
    s = UvicornServer(config)

    if is_temp_server_running():
        # # TODO: redirect temp server logs to a file instead of hidding the logs completely
        # # Redirect stdout and stderr to null
        with (
            open(os.devnull, "w", encoding="utf-8") as f,
            redirect_stdout(f),
            redirect_stderr(f),
        ):
            s.run()
    else:
        s.run()
