# SPDX-License-Identifier: Apache-2.0

# Standard
from contextlib import redirect_stderr, redirect_stdout
from time import sleep
from typing import Optional, cast
import logging
import multiprocessing
import os
import pathlib

# Third Party
from llama_cpp import llama_chat_format, llama_token_get_text
from llama_cpp.server.app import create_app
from llama_cpp.server.model import LlamaProxy
from llama_cpp.server.settings import Settings
import httpx
import llama_cpp.server.app as llama_app

# Local
from ...client import check_api_base
from ...configuration import get_api_base
from .backends import (
    API_ROOT_WELCOME_MESSAGE,
    CHAT_TEMPLATE_AUTO,
    CHAT_TEMPLATE_TOKENIZER,
    BackendServer,
    ServerException,
    UvicornServer,
    free_tcp_ipv4_port,
    get_model_template,
    get_uvicorn_config,
    is_temp_server_running,
    verify_template_exists,
)

logger = logging.getLogger(__name__)


class Server(BackendServer):
    def __init__(
        self,
        model_path: pathlib.Path,
        model_family: str,
        chat_template: str,
        api_base: str,
        host: str,
        port: int,
        gpu_layers: int,
        max_ctx_size: int,
        num_threads: Optional[int],
    ):
        super().__init__(model_family, model_path, chat_template, api_base, host, port)
        self.gpu_layers = gpu_layers
        self.max_ctx_size = max_ctx_size
        self.num_threads = num_threads
        self.queue: Optional[multiprocessing.Queue] = None
        self.process: multiprocessing.Process | None = None

    def run(self):
        """Start an OpenAI-compatible server with llama-cpp"""
        try:
            server(
                model_path=self.model_path,
                chat_template=self.chat_template,
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

        server_process = mpctx.Process(
            target=server,
            kwargs={
                "model_path": self.model_path,
                "chat_template": self.chat_template,
                "gpu_layers": self.gpu_layers,
                "max_ctx_size": self.max_ctx_size,
                "model_family": self.model_family,
                "port": port,
                "host": self.host,
                "queue": self.queue,
            },
        )

        return server_process

    def run_detached(
        self, http_client: httpx.Client | None = None, background: bool = True
    ) -> str:
        logger.info(f"Trying to connect to model server at {self.api_base}")
        if check_api_base(self.api_base, http_client):
            return self.api_base
        try:
            self.port = free_tcp_ipv4_port(self.host)
            # start new server
            self.api_base = str(get_api_base(f"{self.host}:{self.port}"))
            logger.debug(f"Starting a temporary server at {self.api_base}...")
            self.process = self.create_server_process(self.port)
            self.process.start()

            # in case the server takes some time to fail we wait a bit
            logger.debug("Waiting for the server to start...")
            count = 0
            while self.process.is_alive():
                sleep(0.1)
                if check_api_base(self.api_base, http_client):
                    break
                if count > 50:
                    logger.error("failed to reach the API server")
                    break
                count += 1

            logger.debug("Server started.")

            # if the queue is not empty it means the server failed to start
            if self.queue is not None and not self.queue.empty():
                # pylint: disable=raise-missing-from
                raise self.queue.get()

        except ServerException as exc:
            self.shutdown()
            raise exc
        return self.api_base

    def shutdown(self):
        """Stop the server process and close the queue."""

        super().shutdown()

        if self.process and self.queue:
            self.process.terminate()
            self.process.join(timeout=30)
            self.queue.close()
            self.queue.join_thread()


def server(
    model_path: pathlib.Path,
    chat_template: str,
    gpu_layers: int,
    max_ctx_size: int,
    model_family: str,
    threads=None,
    host: str = "localhost",
    port: int = 8000,
    queue: Optional[multiprocessing.Queue] = None,
):
    """Start OpenAI-compatible server"""
    settings = Settings(
        host=host,
        port=port,
        model=model_path.as_posix(),
        n_ctx=max_ctx_size,
        n_gpu_layers=gpu_layers,
        verbose=logger.isEnabledFor(logging.DEBUG),
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

    # Update chat template if necessary
    augment_chat_template(chat_template, model_family, model_path, queue)

    logger.info("Starting server process, press CTRL+C to shutdown server...")
    logger.info(
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


def augment_chat_template(
    chat_template: str,
    model_family: str,
    model_path: pathlib.Path,
    queue: Optional[multiprocessing.Queue],
):
    # chat template takes the format ('auto' | 'tokenizer' | a filesystem path to a file)
    if chat_template == CHAT_TEMPLATE_TOKENIZER:
        # llama_cpp_python calculates from the tokenizer config on load, so
        # nothing to do
        return

    try:
        if chat_template == CHAT_TEMPLATE_AUTO:
            # Currently "auto" maps to replacing with ilab stored templates
            template, eos_token, bos_token = get_model_template(
                model_family, model_path
            )
        else:
            # In this case, the template is a path to a file; attempt to load it
            template = load_template(chat_template)
            eos_token = None
            bos_token = None

        logger.info("Replacing chat template:\n %s", template)

        for proxy in llama_app.get_llama_proxy():
            proxy().chat_handler = llama_chat_format.Jinja2ChatFormatter(
                template=template,
                # Use the model defined eos and bos if either is not
                # defined (when a custom template is used)
                eos_token=resolve_token_eos(eos_token, proxy),
                bos_token=resolve_token_bos(bos_token, proxy),
            ).to_chat_handler()
    # pylint: disable=broad-exception-caught
    except Exception as exc:
        if queue:
            queue.put(exc)
            queue.close()
            queue.join_thread()
            return
        raise ServerException(f"failed creating the server application: {exc}") from exc


def resolve_token_eos(eos_token: Optional[str], proxy: LlamaProxy) -> str:
    if eos_token is not None:
        return eos_token

    return resolve_token(proxy, proxy().token_eos())


def resolve_token_bos(bos_token: Optional[str], proxy: LlamaProxy) -> str:
    if bos_token is not None:
        return bos_token

    return resolve_token(proxy, proxy().token_bos())


def resolve_token(proxy: LlamaProxy, token: int) -> str:
    result = llama_token_get_text(proxy().model, token).decode("utf-8")
    return cast(str, result)


def load_template(template_path: str) -> str:
    path = pathlib.Path(template_path)
    verify_template_exists(path)

    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        raise IOError("Error reading chat template file contents: {}".format(e)) from e
