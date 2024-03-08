# Third Party
from openai import OpenAI, OpenAIError


class ClientException(Exception):
    """An exception raised when invoking client operations."""


def list_models(api_base):
    """List models from OpenAI-compatible server"""
    try:
        client = OpenAI(base_url=api_base, api_key="no_api_key")
        return client.models.list()
    except OpenAIError as exc:
        raise ClientException(f"Connection Error {exc}") from exc
