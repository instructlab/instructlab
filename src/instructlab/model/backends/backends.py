# SPDX-License-Identifier: Apache-2.0

# Standard
import abc
import logging
import pathlib


class BackendServer(abc.ABC):
    """Base class for a serving backend"""

    def __init__(
        self,
        logger: logging.Logger,
        model_path: pathlib.Path,
        host: str,
        port: int,
        **kwargs,
    ) -> None:
        self.logger = logger
        self.model_path = model_path
        self.host = host
        self.port = port

    @abc.abstractmethod
    def run(self):
        """Run serving backend"""

    @abc.abstractmethod
    def shutdown(self):
        """Shutdown serving backend"""
