# SPDX-License-Identifier: Apache-2.0
# pylint: disable=duplicate-code

# Standard
import logging

# Third Party
from openai import OpenAI, OpenAIError
import httpx

# Local
from .configuration import DEFAULTS

logger = logging.getLogger(__name__)


class ClientException(Exception):
    """An exception raised when invoking client operations."""


class ModelCheckException(Exception):
    """An exception raised when checking model."""


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
        logger.info(f"Trying to connect to model server at {api_base}")
        list_models(api_base=api_base, http_client=http_client)
        return True
    except ClientException:
        return False


def check_model(model, api_base: str, endpoint_url=None, http_client=None):
    """Check if right model is selected or served."""
    if endpoint_url:
        api_base = endpoint_url
    model_list = list_models(
        api_base=api_base,
        http_client=http_client,
    )
    if len(model_list.data) == 0:
        raise ModelCheckException("No model is served")

    model_ids = [m.id for m in model_list.data]
    if model not in model_ids:
        if endpoint_url:
            msg = f"Model {model} is not served by the endpoint url {endpoint_url}. These are the served models: {model_ids}. Use '--model' to select right model to use."
        else:
            msg = f"Model {model} is not served by the server. These are the served models: {model_ids}. Use '--model' to select right model to use. Serve the model {model} by running 'ilab serve --model-path {model}' or update the configuration file"
        raise ModelCheckException(msg)
