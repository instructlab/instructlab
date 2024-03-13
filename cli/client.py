# SPDX-FileCopyrightText: The InstructLab Authors
# SPDX-License-Identifier: Apache-2.0

# Third Party
from openai import OpenAI, OpenAIError

# Local
from .config import DEFAULT_API_KEY


class ClientException(Exception):
    """An exception raised when invoking client operations."""


def list_models(api_base, api_key=DEFAULT_API_KEY):
    """List models from OpenAI-compatible server"""
    try:
        client = OpenAI(base_url=api_base, api_key=api_key)
        return client.models.list()
    except OpenAIError as exc:
        raise ClientException(f"Connection Error {exc}") from exc
