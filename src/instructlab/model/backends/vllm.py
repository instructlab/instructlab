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
        super().__init__(logger, api_base, model_path, host, port)
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
            self.logger.info(f"vLLM server terminated by keyboard")
        except BaseException:
            self.shutdown()
            self.logger.exception(f"vLLM server terminated")

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
        self, tls_insecure, tls_client_cert, tls_client_key, tls_client_passwd
    ):
        try:
            _, vllm_server_process, api_base = ensure_server(
                logger=self.logger,
                backend=VLLM,
                api_base=self.api_base,
                host=self.host,
                port=self.port,
                tls_insecure=tls_insecure,
                tls_client_cert=tls_client_cert,
                tls_client_key=tls_client_key,
                tls_client_passwd=tls_client_passwd,
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
        vllm_args += f" --host {host}"

    if "--port" not in vllm_args:
        vllm_args += f" --port {port}"

    if "--model" not in vllm_args:
        vllm_args += f" --model {model_path.as_posix()}"

    vllm_cmd.extend(vllm_args.split())

    logger.debug(f"vLLM serving command is: {vllm_cmd}")
    if background:
        vllm_process = subprocess.Popen(
            args=vllm_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    else:
        vllm_process = subprocess.Popen(args=vllm_cmd)

    api_base = get_api_base(f"{host}:{port}")
    logger.info(f"vLLM starting up on pid {vllm_process.pid} at {api_base}")

    return vllm_process
