# SPDX-License-Identifier: Apache-2.0
# pylint: disable=duplicate-code

# Standard
from typing import Optional

# Third Party
from openai import OpenAI, OpenAIError
import httpx

# Local
from .config import DEFAULT_API_KEY, DEFAULT_CONNECTION_TIMEOUT


class ClientException(Exception):
    """An exception raised when invoking client operations."""


def list_models(
    api_base,
    tls_insecure,
    api_key=DEFAULT_API_KEY,
    tls_client_cert: Optional[str] = None,
    tls_client_key: Optional[str] = None,
    tls_client_passwd: Optional[str] = None,
):
    """List models from OpenAI-compatible server"""
    try:
        orig_cert = (tls_client_cert, tls_client_key, tls_client_passwd)
        cert = tuple(item for item in orig_cert if item)
        verify = not tls_insecure
        client = OpenAI(
            base_url=api_base,
            api_key=api_key,
            timeout=DEFAULT_CONNECTION_TIMEOUT,
            http_client=httpx.Client(cert=cert, verify=verify),
        )
        return client.models.list()
    except OpenAIError as exc:
        raise ClientException(f"Connection Error {exc}") from exc
