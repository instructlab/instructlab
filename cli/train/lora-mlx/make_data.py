# SPDX-FileCopyrightText: The InstructLab Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
import json

# Third Party
import click

SYS_PROMPT = "You are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior."


def format_text(obj):
    return f"""\
<|system|>
{obj['system']}
<|user|>
{obj['user']}
<|assistant|>
{obj['assistant']}<|endoftext|>\
"""


@click.command()
@click.option("--data-dir")
@click.option("--is-shiv", is_flag=True)
def make_data(data_dir, is_shiv):
    if not is_shiv:
        # This branch uses data from `lab generate`
        # train_gen.jsonl and test_gen.jsonl are the two files produced by `lab generate`
        for fn in [f"{data_dir}/train_gen.jsonl", f"{data_dir}/test_gen.jsonl"]:

            # Load the JSON Lines file
            with open(fn, "r") as f:
                data = [json.loads(line) for line in f]

            # Add the "text" field with value "x" to each object
            data_new = []
            for obj in data:
                obj_new = {}
                obj_new["system"] = SYS_PROMPT
                obj_new["user"] = obj["user"]
                obj_new["assistant"] = obj["assistant"]
                data_new.append(obj_new | {"text": format_text(obj_new)})

            # Save the modified objects back to the JSON Lines file
            if "train_gen" in fn:
                n = len(data_new) // 10 * 8
                with open(f"{data_dir}/train.jsonl", "w") as f:
                    for obj in data_new[:n]:
                        f.write(json.dumps(obj) + "\n")
                with open(f"{data_dir}/valid.jsonl", "w") as f:
                    for obj in data_new[n:]:
                        f.write(json.dumps(obj) + "\n")
            if "test_gen" in fn:
                with open(f"{data_dir}/test.jsonl", "w") as f:
                    for obj in data_new:
                        f.write(json.dumps(obj) + "\n")
    else:
        # This branch is to use Shiv generated data
        # You can ignore for now
        fn = f"{data_dir}/raw.jsonl"

        # Load the JSON Lines file
        with open(fn, "r") as f:
            data = [json.loads(line) for line in f]

        # Add the "text" field with value "x" to each object
        data_new = []
        for obj in data:
            obj_new = {}
            obj_new["system"] = SYS_PROMPT
            obj_new["user"] = obj["inputs"]
            obj_new["assistant"] = obj["targets"]
            data_new.append(obj_new | {"text": format_text(obj_new)})

        # Save the modified objects back to the JSON Lines file
        n = len(data_new) // 10 * 7
        m = len(data_new) // 10 * 2 + n
        with open(f"{data_dir}/train.jsonl", "w") as f:
            for obj in data_new[:n]:
                f.write(json.dumps(obj) + "\n")
        with open(f"{data_dir}/valid.jsonl", "w") as f:
            for obj in data_new[n:m]:
                f.write(json.dumps(obj) + "\n")
        with open(f"{data_dir}/test.jsonl", "w") as f:
            for obj in data_new[:m]:
                f.write(json.dumps(obj) + "\n")


if __name__ == "__main__":
    make_data()
