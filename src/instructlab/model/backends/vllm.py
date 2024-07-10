# SPDX-License-Identifier: Apache-2.0

# Standard
import logging
import os
import pathlib
import subprocess
import sys
import time
import typing

# Third Party
import httpx

# Local
from ...configuration import get_api_base
from .backends import (
    VLLM,
    BackendServer,
    ServerException,
    ensure_server,
    shutdown_process,
)

logger = logging.getLogger(__name__)


class Server(BackendServer):
    def __init__(
        self,
        api_base: str,
        model_path: pathlib.Path,
        host: str,
        port: int,
        served_model_name: str,
        max_model_len: int,
        tensor_parallel_size: int,
        max_parallel_loading_workers: int | None,
        device: str,
        vllm_additional_args: typing.Iterable[str] | None = (),
    ):
        super().__init__(model_path, api_base, host, port)
        self.api_base = api_base
        self.model_path = model_path
        self.vllm_additional_args: list[str]
        self.vllm_additional_args = (
            list(vllm_additional_args) if vllm_additional_args is not None else []
        )
        self.served_model_name = served_model_name
        self.device = device
        self.tensor_parallel_size = tensor_parallel_size
        self.max_parallel_loading_workers = max_parallel_loading_workers
        self.max_model_len = max_model_len
        self.process: subprocess.Popen | None = None

    def run(self):
        self.process = run_vllm(
            self.host,
            self.port,
            self.model_path,
            self.served_model_name,
            self.device,
            self.max_model_len,
            self.tensor_parallel_size,
            self.max_parallel_loading_workers,
            self.vllm_additional_args,
            background=False,
        )

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.shutdown()
            self.logger.info("vLLM server terminated by keyboard")
        # pylint: disable=broad-exception-caught
        except BaseException:
            self.shutdown()
            self.logger.exception("vLLM server terminated")

    def create_server_process(self, port: int) -> subprocess.Popen:
        server_process = run_vllm(
            self.host,
            port,
            self.model_path,
            self.served_model_name,
            self.device,
            self.max_model_len,
            self.tensor_parallel_size,
            self.max_parallel_loading_workers,
            self.vllm_additional_args,
            background=True,
        )
        return server_process

    def run_detached(self, http_client: httpx.Client | None = None) -> str:
        try:
            _, vllm_server_process, api_base = ensure_server(
                backend=VLLM,
                api_base=self.api_base,
                http_client=http_client,
                host=self.host,
                port=self.port,
                server_process_func=self.create_server_process,
            )
            self.process = vllm_server_process or self.process
            self.api_base = api_base or self.api_base
        except ServerException as exc:
            raise exc
        except SystemExit as exc:
            raise exc
        return self.api_base

    def shutdown(self):
        """Shutdown vLLM server"""
        # Needed when a temporary server is started
        if self.process is not None:
            shutdown_process(self.process, 20)


def run_vllm(
    host: str,
    port: int,
    model_path: pathlib.Path,
    served_model_name: str,
    device: str,
    max_model_len: int,
    tensor_parallel_size: int,
    max_parallel_loading_workers: int | None,
    vllm_additional_args: list[str],
    background: bool,
) -> subprocess.Popen:
    """
    Start an OpenAI-compatible server with vLLM.

    Args:
        host       (str):             host to run server on
        port       (str):             port to run server on
        model_path (Path):            The path to the model file.
        vllm_additional_args  (list of str):     Specific arguments to pass into vllm.
                                      Example: ["--dtype", "auto", "--enable-lora"]
        background (bool):            Whether the stdout and stderr vLLM should be sent to /dev/null (True)
                                      or stay in the foreground(False).
    Returns:
        vllm_process (subprocess.Popen): process of the vllm server
    """
    vllm_process = None
    vllm_cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]

    # TODO: there should really be a better way to do this, and there probably is
    vllm_cmd.extend(["--host", host])
    vllm_cmd.extend(["--port", str(port)])
    vllm_cmd.extend(["--served-model-name", served_model_name])
    vllm_cmd.extend(["--max-model-len", str(max_model_len)])
    vllm_cmd.extend(["--device", device])
    vllm_cmd.extend(["--tensor-parallel-size", str(tensor_parallel_size)])
    vllm_cmd.extend(
        ["--max-parallel-loading-workers", str(max_parallel_loading_workers)]
    )

    if "--model" not in vllm_additional_args:
        vllm_cmd.extend(["--model", os.fspath(model_path)])

    vllm_cmd.extend(vllm_additional_args)

    logger.debug(f"vLLM serving command is: {vllm_cmd}")
    if background:
        vllm_process = subprocess.Popen(
            args=vllm_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    else:
        # pylint: disable=consider-using-with
        vllm_process = subprocess.Popen(args=vllm_cmd)

    api_base = get_api_base(host, str(port))
    logger.info(f"vLLM starting up on pid {vllm_process.pid} at {api_base}")

    return vllm_process
