# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
from time import time
from typing import Any, Dict
import logging
import os

# Third Party
from datasets import load_dataset
from openai import OpenAI, Stream
import click

# First Party
from instructlab.configuration import DEFAULTS

# Local
from ..utils import get_sysprompt, http_client
from .backends import backends

logger = logging.getLogger(__name__)


def response(client, user: str, create_params: dict):
    # https://platform.openai.com/docs/api-reference/chat/create
    logger.debug("sending %d bytes", len(user))
    start_time = time()
    resp = client.chat.completions.create(
        **create_params,
        messages=[
            {"role": "system", "content": get_sysprompt()},
            {"role": "user", "content": user},
        ],
    )
    if isinstance(resp, Stream):
        # https://platform.openai.com/docs/api-reference/chat/streaming
        resp = ""
        for r in resp:
            if c := r.choices[0].delta.content:
                print(c, end="", flush=True)
                resp += c
        print()
        return resp
    # https://platform.openai.com/docs/api-reference/chat/object
    c = resp.choices[0].message.content
    logger.debug("received %d bytes, elapsed %.1fs", len(c), time() - start_time)
    return c


def test_model(ctx, res, ds, model: Path, create_params: dict):
    logger.debug("%s", model)
    create_params["model"] = str(model)  # mandatory, but not used
    # TODO: this code block is replicated again. Refactor to a common function.
    backend_instance = None
    try:
        ctx.obj.config.serve.llama_cpp.llm_family = ctx.params["model_family"]
        backend_instance = backends.select_backend(
            ctx.obj.config.serve, model_path=model
        )
        try:
            api_base = backend_instance.run_detached(http_client(ctx.params))
        except Exception as exc:
            click.secho(f"Failed to start server: {exc}", fg="red")
            raise click.exceptions.Exit(1)
        api_base = api_base or ctx.obj.config.serve.api_base()
        logger.debug("api_base=%s", api_base)
        client = OpenAI(
            base_url=api_base,
            api_key=ctx.params["api_key"],
        )
        for d in ds:
            res[d["user"]][str(model)] = response(client, d["user"], create_params)
    except Exception as exc:
        raise exc
    finally:
        if backend_instance:
            backend_instance.shutdown()


def linux_test(
    ctx: click.Context,
    test_file: Path,
    models=None,
    create_params=None,
) -> Dict[str, Any]:
    # linux_test
    logger.debug("test_file=%s", test_file)
    if not models:
        models = [DEFAULTS.DEFAULT_MODEL]
    if not create_params:
        create_params = {}

    # https://platform.openai.com/docs/api-reference/chat/create
    create_params["temperature"] = 0  # for more reproducible results
    # create_params["stream"] = True  # optional, useful for interactive debugging

    ds = load_dataset("json", data_files=os.fspath(test_file), split="train")
    res: dict[str, Any] = {}
    # intentionally not using collections.defaultdict to avoid dependency
    for d in ds:
        res[d["user"]] = {}
    for m in models:
        test_model(ctx, res, ds, m, create_params)
    return res
