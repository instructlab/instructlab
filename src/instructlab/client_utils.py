# SPDX-License-Identifier: Apache-2.0
# pylint: disable=duplicate-code

# Standard

# Third Party
from openai import OpenAI, OpenAIError
import httpx

# Local
from .configuration import DEFAULTS


class ClientException(Exception):
    """An exception raised when invoking client operations."""


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
