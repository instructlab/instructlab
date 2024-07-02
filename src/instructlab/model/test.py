# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
import json
import logging
import os

# Third Party
import click

# First Party
from instructlab import utils

# Local
from .. import configuration as cfg

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--data-dir",
    help="Base directory where data is stored.",
    default="./taxonomy_data",
    show_default=True,
)
# for macOS:
@click.option(
    "--model-dir",
    help="Base directory where model is stored.",
    default="instructlab-merlinite-7b-lab-mlx-q",
    show_default=True,
)
@click.option(
    "--adapter-file",
    help="LoRA adapter to use for test. Set to 'None' to force only testing behavior from before training.",
    default="auto",
    show_default=True,
)
# for Linux:
@click.option(
    "-m",
    "--model",
    default=cfg.DEFAULT_MODEL,
    show_default=True,
    help="Base model name to test on Linux",
)
@click.option(
    "-t",
    "--test_file",
    help="Test data file",
    type=click.Path(),
)
@click.option(
    "--api-key",
    type=click.STRING,
    default=cfg.DEFAULT_API_KEY,  # Note: do not expose default API key
    help="API key for API endpoint. [default: cfg.DEFAULT_API_KEY]",
)
@click.option(
    "--model-family",
    help="Force model family to use when picking a generation template",
)
@click.pass_context
@utils.display_params
# pylint: disable=function-redefined
def test(
    ctx,
    data_dir,
    # for macOS:
    model_dir,
    adapter_file,
    # for Linux:
    model,  # pylint: disable=unused-argument
    test_file,
    api_key,  # pylint: disable=unused-argument
    model_family,  # pylint: disable=unused-argument
):
    """Runs basic test to ensure model correctness"""
    if utils.is_macos_with_m_chip():
        # pylint: disable=C0415
        # Local
        from ..train.lora_mlx.lora import load_and_train

        if adapter_file == "auto":
            adapter_file = os.path.join(model_dir, "adapters.npz")
        elif adapter_file.lower() == "none":
            adapter_file = None

        adapter_file_exists = adapter_file and os.path.exists(adapter_file)
        if adapter_file and not adapter_file_exists:
            print(
                "NOTE: Adapter file does not exist. Testing behavior before training only. - %s"
                % adapter_file
            )

        # Load the JSON Lines file
        test_data_dir = f"{data_dir}/test.jsonl"
        if not os.path.exists(test_data_dir):
            click.secho(
                f"'{test_data_dir}' not such file or directory. Did you run 'ilab model train'?",
                fg="red",
            )
            raise click.exceptions.Exit(1)
        with open(test_data_dir, "r", encoding="utf-8") as f:
            test_data = [json.loads(line) for line in f]

        print("system prompt:", utils.get_sysprompt())
        for idx, example in enumerate(test_data):
            system = example["system"]
            user = example["user"]
            print("[{}]\n user prompt: {}".format(idx + 1, user))
            prompt = f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>"
            print("expected output:", example["assistant"])

            print("\n-----model output BEFORE training----:\n")
            load_and_train(
                model=model_dir, no_adapter=True, max_tokens=100, prompt=prompt
            )

            if adapter_file_exists:
                print("\n-----model output AFTER training----:\n")
                load_and_train(
                    model=model_dir,
                    adapter_file=adapter_file,
                    max_tokens=100,
                    prompt=prompt,
                )
    else:
        logger.debug("")
        # pylint: disable=import-outside-toplevel
        # Local
        from .linux_test import linux_test

        if not test_file:
            test_file = sorted(
                Path(ctx.obj.config.generate.output_dir).glob("test_*"),
                key=os.path.getmtime,
                reverse=True,
            )[0]
        logger.debug("test_file=%s", test_file)
        try:
            res = linux_test(
                ctx,
                test_file,
                models=[model, Path("models") / "ggml-model-f16.gguf"],
                create_params={"max_tokens": 100},
            )
            for u in res.items():
                # print in markdown format
                print()
                print("###", u)
                for m in res[u].items():
                    print()
                    print(f"{m}: {res[u][m]}")
                    print()
        except Exception as exc:
            click.secho(
                f"Tesing models failed with the following error: {exc}", fg="red"
            )
            raise click.exceptions.Exit(1)
