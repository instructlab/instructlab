# SPDX-License-Identifier: Apache-2.0

# Standard
from contextlib import redirect_stderr, redirect_stdout
from typing import Optional
import logging
import multiprocessing
import os
import pathlib

# Third Party
from llama_cpp import llama_chat_format
from llama_cpp.server.app import create_app
from llama_cpp.server.settings import Settings
import httpx
import llama_cpp.server.app as llama_app

# Local
from ...configuration import get_model_family
from .backends import (
    API_ROOT_WELCOME_MESSAGE,
    LLAMA_CPP,
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
        model_path: pathlib.Path,
        model_family: str,
        api_base: str,
        host: str,
        port: int,
        gpu_layers: int,
        max_ctx_size: int,
        num_threads: Optional[int],
    ):
        super().__init__(logger, model_path, api_base, host, port)
        self.model_family = model_family
        self.gpu_layers = gpu_layers
        self.max_ctx_size = max_ctx_size
        self.num_threads = num_threads
        self.queue: Optional[multiprocessing.Queue] = None
        self.process: multiprocessing.Process | None = None

    def run(self):
        """Start an OpenAI-compatible server with llama-cpp"""
        try:
            server(
                server_logger=self.logger,
                model_path=self.model_path,
                gpu_layers=self.gpu_layers,
                max_ctx_size=self.max_ctx_size,
                model_family=self.model_family,
                threads=self.num_threads,
                host=self.host,
                port=self.port,
            )
        except ServerException as exc:
            raise exc

    def create_server_process(self, port: int) -> multiprocessing.Process:
        mpctx = multiprocessing.get_context(None)
        self.queue = mpctx.Queue()
        host_port = f"{self.host}:{self.port}"
        # create a temporary, throw-away logger
        server_logger = logging.getLogger(host_port)
        server_logger.setLevel(logging.FATAL)

        server_process = mpctx.Process(
            target=server,
            kwargs={
                "server_logger": server_logger,
                "model_path": self.model_path,
                "gpu_layers": self.gpu_layers,
                "max_ctx_size": self.max_ctx_size,
                "model_family": self.model_family,
                "port": port,
                "host": self.host,
                "queue": self.queue,
            },
        )

        return server_process

    def run_detached(self, http_client: httpx.Client | None = None) -> str:
        try:
            llama_cpp_server_process, _, api_base = ensure_server(
                logger=self.logger,
                backend=LLAMA_CPP,
                api_base=self.api_base,
                http_client=http_client,
                host=self.host,
                port=self.port,
                queue=self.queue,
                server_process_func=self.create_server_process,
            )
            self.process = llama_cpp_server_process or self.process
            self.api_base = api_base or self.api_base
        except ServerException as exc:
            raise exc
        return self.api_base

    def shutdown(self):
        """Stop the server process and close the queue."""
        if self.process and self.queue:
            self.process.terminate()
            self.process.join(timeout=30)
            self.queue.close()
            self.queue.join_thread()


def server(
    server_logger: logging.Logger,
    model_path: pathlib.Path,
    gpu_layers: int,
    max_ctx_size: int,
    model_family: str,
    threads=None,
    host: str = "localhost",
    port: int = 8000,
    queue=None,
):
    """Start OpenAI-compatible server"""
    settings = Settings(
        host=host,
        port=port,
        model=model_path.as_posix(),
        n_ctx=max_ctx_size,
        n_gpu_layers=gpu_layers,
        verbose=server_logger.isEnabledFor(logging.DEBUG),
    )
    if threads is not None:
        settings.n_threads = threads
    try:
        app = create_app(settings=settings)

        @app.get("/")
        def read_root():
            return {"message": API_ROOT_WELCOME_MESSAGE}
    except ValueError as exc:
        if queue:
            queue.put(exc)
            queue.close()
            queue.join_thread()
            return
        raise ServerException(f"failed creating the server application: {exc}") from exc

    template = ""
    eos_token = "<|endoftext|>"
    bos_token = ""
    for template_dict in templates:
        if template_dict["model"] == get_model_family(model_family, model_path):
            template = template_dict["template"]
            if template_dict["model"] == "mixtral":
                eos_token = "</s>"
                bos_token = "<s>"
    try:
        for proxy in llama_app.get_llama_proxy():
            proxy().chat_handler = llama_chat_format.Jinja2ChatFormatter(
                template=template,
                eos_token=eos_token,
                bos_token=bos_token,
            ).to_chat_handler()
    # pylint: disable=broad-exception-caught
    except Exception as exc:
        if queue:
            queue.put(exc)
            queue.close()
            queue.join_thread()
            return
        raise ServerException(f"failed creating the server application: {exc}") from exc

    server_logger.info("Starting server process, press CTRL+C to shutdown server...")
    server_logger.info(
        f"After application startup complete see http://{host}:{port}/docs for API."
    )

    config = get_uvicorn_config(
        app=app,
        host=host,
        port=port,
    )
    s = UvicornServer(config)

    # If this is not the main process, this is the temp server process that ran in the background
    # after `ilab model chat` was executed.
    # In this case, we want to redirect stdout to null to avoid cluttering the chat with messages
    # returned by the server.
    if is_temp_server_running():
        # TODO: redirect temp server logs to a file instead of hidding the logs completely
        # Redirect stdout and stderr to null
        with (
            open(os.devnull, "w", encoding="utf-8") as f,
            redirect_stdout(f),
            redirect_stderr(f),
        ):
            s.run()
    else:
        s.run()

    if queue:
        queue.close()
        queue.join_thread()
