# SPDX-License-Identifier: Apache-2.0

# Standard
from time import sleep
import logging
import pathlib
import subprocess
import sys

# Local
from ...configuration import get_api_base
from .backends import (
    VLLM,
    BackendServer,
    ServerException,
    ensure_server,
    shutdown_process,
)


class Server(BackendServer):
    def __init__(
        self,
        logger: logging.Logger,
        api_base: str,
        model_path: pathlib.Path,
        host: str,
        port: int,
        vllm_args: str = "",
    ):
        super().__init__(logger, model_path, api_base, host, port)
        self.api_base = api_base
        self.model_path = model_path
        self.vllm_args = vllm_args
        self.process = None

    def run(self):
        self.process = run_vllm(
            self.logger,
            self.host,
            self.port,
            self.model_path,
            self.vllm_args,
            background=False,
        )

        try:
            while True:
                sleep(1)
        except KeyboardInterrupt:
            self.shutdown()
            self.logger.info("vLLM server terminated by keyboard")
        # pylint: disable=broad-exception-caught
        except BaseException:
            self.shutdown()
            self.logger.exception("vLLM server terminated")

    def create_server_process(self, port: str) -> subprocess.Popen:
        server_process = run_vllm(
            self.logger,
            self.host,
            port,
            self.model_path,
            self.vllm_args,
            background=True,
        )
        return server_process

    def run_detached(
        self,
        http_client,
    ):
        try:
            _, vllm_server_process, api_base = ensure_server(
                logger=self.logger,
                backend=VLLM,
                api_base=self.api_base,
                http_client=http_client,
                host=self.host,
                port=self.port,
                server_process_func=self.create_server_process,
            )
            self.process = vllm_server_process
            self.api_base = api_base
        except ServerException as exc:
            raise exc
        except SystemExit as exc:
            raise exc

    def shutdown(self):
        """Shutdown vLLM server"""
        # Needed when a temporary server is started
        if self.process is not None:
            shutdown_process(self.process, 20)


def run_vllm(
    logger: logging.Logger,
    host: str,
    port: str,
    model_path: pathlib.Path,
    vllm_args: str,
    background: bool,
) -> subprocess.Popen:
    """
    Start an OpenAI-compatible server with vLLM.

    Args:
        logger     (logging.Logger):  logger for info and debugging
        host       (str):             host to run server on
        port       (str):             port to run server on
        model_path (Path):            The path to the model file.
        vllm_args  (str):             Specific arguments to pass into vllm. Example: "--dtype auto --enable-lora",
        background (bool):            Whether the stdout and stderr vLLM should be sent to /dev/null (True)
                                      or stay in the foreground(False).
    Returns:
        vllm_process (subprocess.Popen): process of the vllm server
    """
    vllm_process = None
    vllm_cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]

    if "--host" not in vllm_args:
        vllm_args = f"{vllm_args} --host {host}"

    if "--port" not in vllm_args:
        vllm_args = f"{vllm_args} --port {port}"

    if "--model" not in vllm_args:
        vllm_args = f"{vllm_args} --model {model_path.as_posix()}"

    vllm_cmd.extend(vllm_args.split())

    logger.debug(f"vLLM serving command is: {vllm_cmd}")
    if background:
        vllm_process = subprocess.Popen(
            args=vllm_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    else:
        # pylint: disable=consider-using-with
        vllm_process = subprocess.Popen(args=vllm_cmd)

    api_base = get_api_base(f"{host}:{port}")
    logger.info(f"vLLM starting up on pid {vllm_process.pid} at {api_base}")

    return vllm_process
