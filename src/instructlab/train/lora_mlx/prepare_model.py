# SPDX-License-Identifier: MIT

# Standard
import os

# Third Party
from transformers import AutoModelForCausalLM, AutoTokenizer
import fire


def convert_bin_to_safetensors(input_dir, output_dir):
    input_dir, output_dir = (
        os.path.expanduser(input_dir),
        os.path.expanduser(output_dir),
    )

    model_path = os.path.expanduser(input_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    tokenizer.save_pretrained(output_dir)
    tokenizer.save_vocabulary(output_dir)
    model.save_pretrained(output_dir)

    # save_dict = model.state_dict()
    # torch.save(save_dict, os.path.join(output_dir, "pytorch_model.bin"))
    # model.config.to_json_file(os.path.join(output_dir, "config.json"))


if __name__ == "__main__":
    fire.Fire(convert_bin_to_safetensors)
