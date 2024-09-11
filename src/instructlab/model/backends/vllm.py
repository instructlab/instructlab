# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Optional, Tuple
import json
import logging
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import typing

# Third Party
import httpx

# Local
from ...client import check_api_base
from ...configuration import get_api_base
from .backends import shutdown_process
from .common import (
    CHAT_TEMPLATE_AUTO,
    CHAT_TEMPLATE_TOKENIZER,
    VLLM,
    Closeable,
    ServerException,
    free_tcp_ipv4_port,
    get_model_template,
    safe_close_all,
    verify_template_exists,
)
from .server import BackendServer

logger = logging.getLogger(__name__)


class Server(BackendServer):
    def __init__(
        self,
        api_base: str,
        model_family: str,
        model_path: pathlib.Path,
        chat_template: str,
        host: str,
        port: int,
        background: bool = False,
        vllm_args: typing.Iterable[str] | None = (),
        max_startup_attempts: int | None = None,
    ):
        super().__init__(model_family, model_path, chat_template, api_base, host, port)
        self.api_base = api_base
        self.model_path = model_path
        self.background = background
        self.vllm_args: list[str]
        self.vllm_args = list(vllm_args) if vllm_args is not None else []
        self.process: subprocess.Popen | None = None
        self.max_startup_attempts = max_startup_attempts

    def run(self):
        self.process, files = run_vllm(
            self.host,
            self.port,
            self.model_family,
            self.model_path,
            self.chat_template,
            self.vllm_args,
            self.background,
        )
        self.register_resources(files)

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("vLLM server terminated by keyboard")
        # pylint: disable=broad-exception-caught
        except BaseException:
            logger.exception("vLLM server terminated")
        finally:
            self.shutdown()

    def create_server_process(self, port: int, background: bool) -> subprocess.Popen:
        server_process, files = run_vllm(
            self.host,
            port,
            self.model_family,
            self.model_path,
            self.chat_template,
            self.vllm_args,
            background=background,
        )
        self.register_resources(files)
        return server_process

    def _ensure_server(
        self,
        http_client=None,
        background=True,
        foreground_allowed=False,
    ) -> Tuple[Optional[subprocess.Popen], Optional[str]]:
        """Checks if server is running, if not starts one as a subprocess. Returns the server process
        and the URL where it's available."""

        logger.info(f"Trying to connect to model server at {self.api_base}")
        if check_api_base(self.api_base, http_client):
            return (None, self.api_base)
        port = free_tcp_ipv4_port(self.host)
        logger.debug(f"Using available port {port} for temporary model serving.")

        host_port = f"{self.host}:{port}"
        temp_api_base = get_api_base(host_port)
        vllm_server_process = self.create_server_process(port, background)
        logger.info("Starting a temporary vLLM server at %s", temp_api_base)
        count = 0
        # Each call to check_api_base takes >2s + 2s sleep
        # Default to 120 if not specified (~8 mins of wait time)
        vllm_startup_max_attempts = self.max_startup_attempts or 120
        start_time_secs = time.time()
        while count < vllm_startup_max_attempts:
            count += 1
            # Check if the process is still alive
            if vllm_server_process.poll():
                if foreground_allowed and background:
                    raise ServerException(
                        "vLLM failed to start.  Retry with --enable-serving-output to learn more about the failure."
                    )
                raise ServerException("vLLM failed to start.")
            logger.info(
                "Waiting for the vLLM server to start at %s, this might take a moment... Attempt: %s/%s",
                temp_api_base,
                count,
                vllm_startup_max_attempts,
            )
            if check_api_base(temp_api_base, http_client):
                logger.info("vLLM engine successfully started at %s", temp_api_base)
                break
            if count == vllm_startup_max_attempts:
                logger.info(
                    "Gave up waiting for vLLM server to start at %s after %s attempts",
                    temp_api_base,
                    vllm_startup_max_attempts,
                )
                duration = round(time.time() - start_time_secs, 1)
                shutdown_process(vllm_server_process, 20)
                # pylint: disable=raise-missing-from
                raise ServerException(f"vLLM failed to start up in {duration} seconds")
            time.sleep(2)
        return (vllm_server_process, temp_api_base)

    def run_detached(
        self,
        http_client: httpx.Client | None = None,
        background: bool = True,
        foreground_allowed: bool = False,
        max_startup_retries: int = 0,
    ) -> str:
        for i in range(max_startup_retries + 1):
            try:
                vllm_server_process, api_base = self._ensure_server(
                    http_client=http_client,
                    background=background,
                    foreground_allowed=foreground_allowed,
                )
                self.process = vllm_server_process or self.process
                self.api_base = api_base or self.api_base
                break
            except ServerException as e:
                if i == max_startup_retries:
                    raise e
                logger.info(
                    "vLLM startup failed.  Retrying (%s/%s)",
                    i + 1,
                    max_startup_retries,
                )
                logger.error(e)

        return self.api_base

    def shutdown(self):
        """Shutdown vLLM server"""

        super().shutdown()

        # Needed when a temporary server is started
        if self.process is not None:
            shutdown_process(self.process, 20)
            self.process = None

    def get_backend_type(self):
        return VLLM


def format_template(model_family: str, model_path: pathlib.Path) -> str:
    template, eos_token, bos_token = get_model_template(model_family, model_path)

    prefix = ""
    if eos_token:
        prefix = '{{% set eos_token = "{}" %}}\n'.format(eos_token)
    if bos_token:
        prefix = '{}{{% set bos_token = "{}" %}}\n'.format(prefix, bos_token)

    return prefix + template


def contains_argument(prefix: str, args: typing.Iterable[str]) -> bool:
    # Either --foo value or --foo=value
    return any(s == prefix or s.startswith(prefix + "=") for s in args)


def get_argument(prefix: str, args: typing.List[str]):
    # Return last value in args for either --foo value or --foo=value
    # Returns True if flag --foo is provided with no value

    args_len = len(args)
    # Traverse the args in reverse (args_len-1 to 0)
    for i in range(args_len - 1, -1, -1):
        s = args[i]
        if s == prefix:
            # Case: --foo value or --foo (with no value)
            if i == args_len - 1:
                # --foo is the last entry, must be a flag
                return True
            next_arg = args[i + 1]
            if next_arg.startswith("-"):
                # No value provided, must be a flag
                return True
            # The entry after is the value
            return next_arg
        v = prefix + "="
        if s.startswith(v):
            # Case: --foo=value
            # Return everything after prefix=
            return s[len(v) :]
    return None


def create_tmpfile(data: str):
    # pylint: disable=consider-using-with
    file = tempfile.NamedTemporaryFile()
    file.write(data.encode("utf-8"))
    file.flush()
    return file


def run_vllm(
    host: str,
    port: int,
    model_family: str,
    model_path: pathlib.Path,
    chat_template: str,
    vllm_args: list[str],
    background: bool,
) -> typing.Tuple[subprocess.Popen, list[Closeable]]:
    """
    Start an OpenAI-compatible server with vLLM.

    Args:
        host          (str):          host to run server on
        port          (int):          port to run server on
        model_family  (str):          Family the model belongs to, used with 'auto' chat templates
        model_path    (Path):         The path to the model file
        chat_template (str):          Chat template to use when serving the model
                                         'auto' (default) automatically determines a template\n
                                         'tokenizer' uses the model provided template\n
                                         (path) specifies a template file location to load from.

        vllm_args     (list of str):  Specific arguments to pass into vllm.
                                        Example: ["--dtype", "auto", "--enable-lora"]
        background (bool):            Whether the stdout and stderr vLLM should be sent to /dev/null (True)
                                      or stay in the foreground(False).
    Returns:
        tuple: A tuple containing two values:
            vllm_process (subprocess.Popen): process of the vllm server
            tmp_files: a list of temporary files necessary to launch the process

    """
    vllm_process = None
    vllm_cmd, tmp_files = build_vllm_cmd(
        host, port, model_family, model_path, chat_template, vllm_args
    )

    logger.debug(f"vLLM serving command is: {vllm_cmd}")

    vllm_env = os.environ.copy()
    # Reset vllm logging to the default (enabled)
    vllm_env.pop("VLLM_CONFIGURE_LOGGING", None)

    try:
        # Note: start_new_session=True is needed to create a process group which will later be used on shutdown
        if background:
            vllm_process = subprocess.Popen(
                args=vllm_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=vllm_env,
                start_new_session=True,
            )
        else:
            # pylint: disable=consider-using-with
            vllm_process = subprocess.Popen(
                args=vllm_cmd,
                env=vllm_env,
                start_new_session=True,
            )

        api_base = get_api_base(f"{host}:{port}")
        logger.info("vLLM starting up on pid %s at %s", vllm_process.pid, api_base)

    except:
        safe_close_all(tmp_files)
        raise

    return vllm_process, tmp_files


def is_bnb_quantized(model_path: pathlib.Path) -> bool:
    """
    Check if provided model has quantization config with bitsandbytes specified.

    Args:
        model_path    (Path):         The path to the model files

    Returns:
        bool: Whether or not the model has been quantized via bitsandbytes
    """
    config_json = model_path / "config.json"
    if not model_path.is_dir() or not config_json.is_file():
        return False

    with config_json.open(encoding="utf-8") as f:
        config = json.load(f)

    return bool(
        config.get("quantization_config", {}).get("quant_method", "") == "bitsandbytes"
    )


def build_vllm_cmd(
    host: str,
    port: int,
    model_family: str,
    model_path: pathlib.Path,
    chat_template: str,
    vllm_args: list[str],
) -> typing.Tuple[list[str], list[Closeable]]:
    """
    Build the vLLM command to run the server.

    Args:
        host          (str):          host to run server on
        port          (int):          port to run server on
        model_family  (str):          Family the model belongs to, used with 'auto' chat templates
        model_path    (Path):         The path to the model file
        chat_template (str):          Chat template to use when serving the model
                                         'auto' (default) automatically determines a template\n
                                         'tokenizer' uses the model provided template\n
                                         (path) specifies a template file location to load from.

        vllm_args     (list of str):  Specific arguments to pass into vllm.
                                        Example: ["--dtype", "auto", "--enable-lora"]

    Returns:
        tuple: A tuple containing two values:
            vllm_cmd (list of str): The command to run the vllm server
            tmp_files: a list of temporary files necessary to launch the process
    """
    vllm_cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
    tmp_files = []

    if not contains_argument("--host", vllm_args):
        vllm_cmd.extend(["--host", host])

    if not contains_argument("--port", vllm_args):
        vllm_cmd.extend(["--port", str(port)])

    if not contains_argument("--model", vllm_args):
        vllm_cmd.extend(["--model", os.fspath(model_path)])

    if not contains_argument("--chat-template", vllm_args):
        if chat_template == CHAT_TEMPLATE_AUTO:
            # For auto templates, the build-in in-memory template
            # needs to be written to a temporary file so that vLLM
            # can read it
            template = format_template(model_family, model_path)
            file = create_tmpfile(template)
            tmp_files.append(file)
            vllm_cmd.extend(["--chat-template", file.name])
        elif chat_template != CHAT_TEMPLATE_TOKENIZER:
            # In this case the value represents a template file path
            # Pass it directly to vllm
            path = pathlib.Path(chat_template)
            verify_template_exists(path)
            vllm_cmd.extend(["--chat-template", chat_template])

    # Auto-detect whether model is quantized w/ bitsandbytes and add potentially missing args
    quant_arg_present = contains_argument("--quantization", vllm_args)
    load_arg_present = contains_argument("--load-format", vllm_args)
    eager_arg_present = contains_argument("--enforce-eager", vllm_args)
    if not (quant_arg_present and load_arg_present and eager_arg_present):
        if is_bnb_quantized(model_path):
            if not quant_arg_present:
                vllm_cmd.extend(["--quantization", "bitsandbytes"])
            if not load_arg_present:
                vllm_cmd.extend(["--load-format", "bitsandbytes"])
            # Currently needed to retain generation quality w/ 4-bit bnb + vLLM (bypass graph bug)
            if not eager_arg_present:
                vllm_cmd.append("--enforce-eager")

    # Force multiprocessing for distributed serving, vLLM will try "ray" if it's installed but we do
    # not support it (yet?). We don't install Ray but we might end up running on systems that have it,
    # so let's make sure we use multiprocessing.
    if not contains_argument("--distributed-executor-backend", vllm_args):
        vllm_cmd.extend(["--distributed-executor-backend", "mp"])

    vllm_cmd.extend(vllm_args)

    return vllm_cmd, tmp_files
