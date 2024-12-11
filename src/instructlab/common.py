# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
import json
import struct

DEFAULT_SYS_PROMPT = "I am an advanced AI language model designed to assist you with a wide range of tasks and provide helpful, clear, and accurate responses. My primary role is to serve as a chat assistant, engaging in natural, conversational dialogue, answering questions, generating ideas, and offering support across various topics."
CLI_HELPER_SYS_PROMPT = "You are an expert for command line interface and know all common commands. Answer the command to execute as it without any explanation."


class SupportedModelArchitectures:
    LLAMA = "llama"
    GRANITE = "granite"


# These system prompts are specific to granite models developed by Red Hat and IBM Research
SYSTEM_PROMPTS = {
    SupportedModelArchitectures.LLAMA: "I am, Red Hat® Instruct Model based on Granite 7B, an AI language model developed by Red Hat and IBM Research, based on the Granite-7b-base language model. My primary function is to be a chat assistant.",
    SupportedModelArchitectures.GRANITE: "I am a Red Hat® Instruct Model, an AI language model developed by Red Hat and IBM Research based on the granite-3.0-8b-base model. My primary role is to serve as a chat assistant.",
}


def create_safetensors_model_directory(
    directory_path: Path, model_dir_name="test_namespace/testlab_model"
):
    """Simulate a safetensors model directory"""
    full_directory_path = directory_path / model_dir_name
    full_directory_path.mkdir(parents=True, exist_ok=True)

    json_data = {"key": "value"}
    required_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]

    for file_name in required_files:
        with open(full_directory_path / file_name, "w", encoding="utf-8") as f:
            json.dump(json_data, f)

    safetensors_file = full_directory_path / "test-model.safetensors"
    # Third Party
    from safetensors.torch import save_file
    import torch

    tensors = {
        "tensor1": torch.randn(3, 3),
        "tensor2": torch.randn(5, 5),
    }
    save_file(tensors, safetensors_file)


def create_gguf_file(file_path: Path, gguf_file_name="test-model.gguf"):
    """Simulate a GGUF file"""
    GGUF_MAGIC = 0x46554747

    full_file_path = file_path / gguf_file_name
    full_file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(full_file_path, "wb") as f:
        f.write(struct.pack("<I", GGUF_MAGIC))
