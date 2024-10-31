# SPDX-License-Identifier: Apache-2.0
# pylint: disable=duplicate-code

# Standard
from typing import TypedDict

# Third Party
from openai import OpenAI, OpenAIError
import httpx

# Local
from .configuration import DEFAULTS


class ClientException(Exception):
    """An exception raised when invoking client operations."""


# pylint: disable=redefined-outer-name
def list_models(
    api_base,
    api_key=DEFAULTS.API_KEY,
    http_client=None,
):
    """List models from OpenAI-compatible server"""
    try:
        client = OpenAI(
            base_url=api_base,
            api_key=api_key,
            timeout=DEFAULTS.CONNECTION_TIMEOUT,
            http_client=http_client,
        )
        return client.models.list()
    except OpenAIError as exc:
        raise ClientException(f"Connection Error {exc}") from exc


def check_api_base(api_base: str, http_client: httpx.Client | None = None) -> bool:
    try:
        list_models(api_base=api_base, http_client=http_client)
        return True
    except ClientException:
        return False


class HttpClientParams(TypedDict):
    """
    Types the parameters used when initializing the HTTP client.
    """

    tls_client_cert: str | None
    tls_client_key: str | None
    tls_client_passwd: str | None
    tls_insecure: bool


def get_ssl_cert_config(tls_client_cert, tls_client_key, tls_client_passwd):
    if tls_client_cert:
        return tls_client_cert, tls_client_key, tls_client_passwd


def http_client(params: HttpClientParams):
    return httpx.Client(
        cert=get_ssl_cert_config(
            params.get("tls_client_cert", None),
            params.get("tls_client_key", None),
            params.get("tls_client_passwd", None),
        ),
        verify=not params.get("tls_insecure", True),
    )
