# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
import json
import logging
import os

# Third Party
import click

# First Party
from instructlab import clickext, utils
from instructlab.configuration import DEFAULTS

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--data-dir",
    help="Base directory where data is stored.",
    default=lambda: DEFAULTS.INTERNAL_DIR,
    show_default="Default internal data directory, stored in the instructlab package.",
)
# for macOS:
@click.option(
    "--model-dir",
    help="Base directory where model is stored.",
    default=lambda: DEFAULTS.CHECKPOINTS_DIR,
    show_default="Default instructlab system checkpoints directory.",
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
    default=lambda: DEFAULTS.DEFAULT_MODEL,
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
    default=DEFAULTS.API_KEY,  # Note: do not expose default API key
    help="API key for API endpoint. [default: cfg.DEFAULT_API_KEY]",
)
@click.option(
    "--model-family",
    help="Force model family to use when picking a generation template",
)
@click.pass_context
@clickext.display_params
# pylint: disable=function-redefined
def test(
    ctx,
    data_dir: str,
    # for macOS:
    model_dir: str,
    adapter_file: str,
    # for Linux:
    model: str,  # pylint: disable=unused-argument
    test_file: Path,
    api_key: str,  # pylint: disable=unused-argument
    model_family: str,  # pylint: disable=unused-argument
):
    """Runs basic test to ensure model correctness"""
    if utils.is_macos_with_m_chip():
        # pylint: disable=import-outside-toplevel
        # Local
        from ..train.lora_mlx.lora import load_and_train

        processed_adapter: str | None = adapter_file
        if adapter_file == "auto":
            processed_adapter = os.path.join(model_dir, "adapters.npz")
        elif adapter_file.lower() == "none":
            processed_adapter = None

        adapter_file_exists = processed_adapter and os.path.exists(processed_adapter)
        if processed_adapter and not adapter_file_exists:
            print(
                "NOTE: Adapter file does not exist. Testing behavior before "
                f"training only. - {processed_adapter}"
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
                model=model_dir,
                no_adapter=True,
                max_tokens=100,
                prompt=prompt,
            )

            if adapter_file_exists:
                print("\n-----model output AFTER training----:\n")
                assert processed_adapter is not None
                load_and_train(
                    model=model_dir,
                    adapter_file=processed_adapter,
                    max_tokens=100,
                    prompt=prompt,
                )
    else:
        logger.debug("")
        # pylint: disable=import-outside-toplevel
        # Local
        from .linux_test import linux_test

        if not test_file:
            try:
                test_file = sorted(
                    Path(ctx.obj.config.generate.output_dir).glob("test_*"),
                    key=os.path.getmtime,
                    reverse=True,
                )[0]
            except Exception as exc:
                click.secho("No test files found", fg="red")
                raise click.exceptions.Exit(1) from exc
        try:
            answers = linux_test(
                ctx,
                test_file,
                models=[model, Path(DEFAULTS.CHECKPOINTS_DIR) / "ggml-model-f16.gguf"],
                create_params={"max_tokens": 100},
            )
            for question, models in answers.items():
                # print in markdown format
                print()
                print("###", question)
                for mod, answer in models.items():
                    print()
                    print(f"{mod}: {answer}")
                print()
        except Exception as exc:
            click.secho(
                f"Testing models failed with the following error: {exc}", fg="red"
            )
            raise click.exceptions.Exit(1)
