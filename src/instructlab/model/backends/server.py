# Standard
import abc
import logging
import pathlib
import typing

# Third Party
import httpx

# Local
from .common import CHAT_TEMPLATE_AUTO, Closeable, safe_close_all

logger = logging.getLogger(__name__)


class BackendServer(abc.ABC):
    """Base class for a serving backend"""

    def __init__(
        self,
        model_family: str,
        model_path: pathlib.Path,
        chat_template: str,
        api_base: str,
        host: str,
        port: int,
    ) -> None:
        self.model_family = model_family
        self.model_path = model_path
        self.chat_template = chat_template if chat_template else CHAT_TEMPLATE_AUTO
        self.api_base = api_base
        self.host = host
        self.port = port
        self.resources: list[Closeable] = []

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
