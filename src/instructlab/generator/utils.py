# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Optional, Sequence, Union
import dataclasses
import io
import json
import logging
import os

# Third Party
from openai import OpenAI, OpenAIError
import httpx

# Local
from ..config import DEFAULT_API_KEY, DEFAULT_MODEL_OLD
from ..utils import get_sysprompt

StrOrOpenAIObject = Union[str, object]


class GenerateException(Exception):
    """An exception raised during generate step."""


# pylint: disable=too-many-instance-attributes
@dataclasses.dataclass
class OpenAIDecodingArguments:
    max_tokens: int = 1800
    temperature: float = 0.2
    top_p: float = 1.0
    stream: bool = False
    stop: Optional[Sequence[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logprobs: Optional[int] = None


def openai_completion(
    api_base,
    tls_insecure,
    tls_client_cert,
    tls_client_key,
    tls_client_passwd,
    prompt: str,
    decoding_args: OpenAIDecodingArguments,
    model_name="ggml-merlinite-7b-lab-Q4_K_M",
    api_key=DEFAULT_API_KEY,
    **decoding_kwargs,
) -> Union[
    Union[StrOrOpenAIObject],
    Sequence[StrOrOpenAIObject],
    Sequence[Sequence[StrOrOpenAIObject]],
]:
    """Decode with OpenAI API.

    Args:
        api_base: Endpoint URL where model is hosted
        tls_insecure: Disable TLS verification
        tls_client_cert: Path to the TLS client certificate to use
        tls_client_key: Path to the TLS client key to use
        tls_client_passwd: TLS client certificate password
        prompt: A string
        decoding_args: Decoding arguments.
        model_name: Model name. Can be either in the format of "org/model" or just "model".
        api_key: API key API key for API endpoint where model is hosted
        decoding_kwargs: Extra decoding arguments. Pass in `best_of` and `logit_bias` if needed.

    Returns:
            - a string
    """
    shared_kwargs = {
        "model": model_name,
        **decoding_args.__dict__,
        **decoding_kwargs,
    }

    if not api_key:
        # we need to explicitly set non-empty api-key, to ensure generate
        # connects to our local server
        api_key = "no_api_key"

    # do not pass a lower timeout to this client since generating a dataset takes some time
    # pylint: disable=R0801
    orig_cert = (tls_client_cert, tls_client_key, tls_client_passwd)
    cert = tuple(item for item in orig_cert if item)
    verify = not tls_insecure
    client = OpenAI(
        base_url=api_base,
        api_key=api_key,
        http_client=httpx.Client(cert=cert, verify=verify),
    )

    # ensure the model specified exists on the server. with backends like vllm, this is crucial.
    model_list = client.models.list().data
    model_ids = []
    for model in model_list:
        model_ids.append(model.id)
    if not any(model_name == m for m in model_ids):
        if model_name == DEFAULT_MODEL_OLD:
            logging.info(
                "Model %s is not a full path. Try running ilab init or edit your config to have the full model path for serving, chatting, and generation.",
                model_name,
            )
        raise GenerateException(
            f"Model {model_name} is not served by the server. These are the served models {model_ids}"
        )

    messages = [
        {"role": "system", "content": get_sysprompt()},
        {"role": "user", "content": prompt},
    ]

    # Inference the model
    try:
        response = client.chat.completions.create(
            messages=messages,
            **shared_kwargs,
        )
    except OpenAIError as exc:
        raise GenerateException(
            f"There was a problem connecting to the server {exc}"
        ) from exc

    return response.choices[0].message.content


def _make_w_io_base(f, mode: str):
    # pylint: disable=consider-using-with
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode, encoding="utf-8")
    return f


def _make_r_io_base(f, mode: str):
    # pylint: disable=consider-using-with
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode, encoding="utf-8")
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    with _make_w_io_base(f, mode) as f_:
        if isinstance(obj, (dict, list)):
            json.dump(obj, f_, indent=indent, default=default)
        elif isinstance(obj, str):
            f_.write(obj)
        else:
            raise ValueError(f"Unexpected type: {type(obj)}")


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    with _make_r_io_base(f, mode) as f_:
        return json.load(f_)
