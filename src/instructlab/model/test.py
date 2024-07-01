# SPDX-License-Identifier: Apache-2.0

# Standard
import json
import os

# Third Party
import click

# First Party
from instructlab import utils
from instructlab.configuration import DEFAULTS


@click.command()
@click.option(
    "--data-dir",
    help="Base directory where data is stored.",
    default=lambda: DEFAULTS.INTERNAL_DIR,
    show_default="Default internal data directory, stored in the instructlab package.",
)
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
@utils.display_params
# pylint: disable=function-redefined
def test(data_dir: str, model_dir: str, adapter_file: str):
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
