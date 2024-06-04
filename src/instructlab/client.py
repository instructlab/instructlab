# SPDX-License-Identifier: Apache-2.0
# pylint: disable=duplicate-code

# Standard
from typing import Optional

# Third Party
from openai import OpenAI, OpenAIError
import httpx

# Local
from .config import DEFAULT_API_KEY, DEFAULT_CONNECTION_TIMEOUT
from .utils import get_ssl_cert_config


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
        cert = get_ssl_cert_config(tls_client_cert, tls_client_key, tls_client_passwd)
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
