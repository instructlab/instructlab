# Standard
from dataclasses import dataclass
import abc
import logging
import pathlib
import typing

# Third Party
import httpx

# Local
from ...log import add_file_handler_to_logger
from .common import CHAT_TEMPLATE_AUTO, Closeable, safe_close_all

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    api_base: str
    log_file: typing.Optional[pathlib.Path] = None


class BackendServer(abc.ABC):
    """Base class for a serving backend"""

    def __init__(
        self,
        model_family: str,
        model_path: pathlib.Path,
        chat_template: str,
        host: str,
        port: int,
        config: ServerConfig,
    ) -> None:
        self.model_family = model_family
        self.model_path = model_path
        self.chat_template = chat_template if chat_template else CHAT_TEMPLATE_AUTO
        self.host = host
        self.port = port
        self.config = config
        self.resources: list[Closeable] = []

        # Write logs to a file if log_file is provided
        if self.config.log_file:
            self.setup_logs_to_file(logger)

    def setup_logs_to_file(self, a_logger: logging.Logger) -> None:
        """Write logs to a file if log_file is provided"""
        add_file_handler_to_logger(a_logger, self.config.log_file)

    @abc.abstractmethod
    def run(self):
        """Run serving backend in foreground (ilab model serve)"""

    @abc.abstractmethod
    def run_detached(
        self,
        http_client: httpx.Client | None = None,
        background: bool = True,
        foreground_allowed: bool = False,
        max_startup_retries: int = 0,
    ) -> str:
        """Run serving backend in background ('ilab model chat' when server is not running)"""

    def shutdown(self):
        """Shutdown serving backend"""

        safe_close_all(self.resources)

    def register_resources(self, resources: typing.Iterable[Closeable]) -> None:
        self.resources.extend(resources)

    @abc.abstractmethod
    def get_backend_type(self):
        """Return which type of backend this is, llama-cpp or vllm"""
