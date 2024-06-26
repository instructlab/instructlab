# SPDX-License-Identifier: MIT

# Standard
import json

# Local
from ...utils import get_sysprompt


def format_text(obj):
    return f"""\
<|system|>
{obj['system']}
<|user|>
{obj['user']}
<|assistant|>
{obj['assistant']}<|endoftext|>\
"""


def make_data(data_dir: str, is_shiv: bool = False):
    if not is_shiv:
        # This branch uses data from `lab generate`
        # train_gen.jsonl and test_gen.jsonl are the two files produced by `lab generate`
        for filename in [f"{data_dir}/train_gen.jsonl", f"{data_dir}/test_gen.jsonl"]:
            # Load the JSON Lines file
            with open(filename, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f]

            # Add the "text" field with value "x" to each object
            data_new = []
            for obj in data:
                obj_new = {
                    "system": get_sysprompt(),
                    "user": obj["user"],
                    "assistant": obj["assistant"],
                }
                data_new.append(obj_new | {"text": format_text(obj_new)})

            # Save the modified objects back to the JSON Lines file
            if "train_gen" in filename:
                n = len(data_new) * 8 // 10
                with open(f"{data_dir}/train.jsonl", "w", encoding="utf-8") as f:
                    for obj in data_new[:n]:
                        f.write(json.dumps(obj) + "\n")
                with open(f"{data_dir}/valid.jsonl", "w", encoding="utf-8") as f:
                    for obj in data_new[n:]:
                        f.write(json.dumps(obj) + "\n")
            if "test_gen" in filename:
                with open(f"{data_dir}/test.jsonl", "w", encoding="utf-8") as f:
                    for obj in data_new:
                        f.write(json.dumps(obj) + "\n")
    else:
        # This branch is to use Shiv generated data
        # You can ignore for now
        filename = f"{data_dir}/raw.jsonl"

        # Load the JSON Lines file
        with open(filename, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

        # Add the "text" field with value "x" to each object
        data_new = []
        for obj in data:
            obj_new = {
                "system": get_sysprompt(),
                "user": obj["inputs"],
                "assistant": obj["targets"],
            }
            data_new.append(obj_new | {"text": format_text(obj_new)})

        # Save the modified objects back to the JSON Lines file
        n = len(data_new) // 10 * 7
        m = len(data_new) // 10 * 2 + n
        with open(f"{data_dir}/train.jsonl", "w", encoding="utf-8") as f:
            for obj in data_new[:n]:
                f.write(json.dumps(obj) + "\n")
        with open(f"{data_dir}/valid.jsonl", "w", encoding="utf-8") as f:
            for obj in data_new[n:m]:
                f.write(json.dumps(obj) + "\n")
        with open(f"{data_dir}/test.jsonl", "w", encoding="utf-8") as f:
            for obj in data_new[:m]:
                f.write(json.dumps(obj) + "\n")
