# SPDX-License-Identifier: MIT

# Standard
from pathlib import Path
import os
import sys

# Third Party
import click

# Get the absolute path of the current script
current_script_path = os.path.abspath(__file__)
# Get the parent directory of the current script
parent_directory = os.path.dirname(current_script_path)
# Add the parent directory to sys.path
sys.path.append(parent_directory)

# Third Party
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
import mlx.core as mx
import transformers

# First Party
from instructlab.utils import macos_requirement


def fetch_tokenizer_from_hub(hf_path: str, local_dir: str):
    model_path = snapshot_download(
        repo_id=hf_path,
        local_dir=local_dir,
        allow_patterns=["*.json", "tokenizer.model"],
    )

    config = transformers.AutoConfig.from_pretrained(hf_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        hf_path,
    )
    return config.to_dict(), tokenizer


def make_shards(weights: dict, max_file_size_gibibyte: int = 15):
    max_file_size_bytes = max_file_size_gibibyte << 30
    shards = []
    shard, shard_size = {}, 0
    for k, v in weights.items():
        if shard_size + v.nbytes > max_file_size_bytes:
            shards.append(shard)
            shard, shard_size = {}, 0
        shard[k] = v
        shard_size += v.nbytes
    shards.append(shard)
    return shards


@macos_requirement(echo_func=click.secho, exit_exception=click.exceptions.Exit)
def save_model(save_dir: str, weights):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    dtype = weights[next(iter(weights.keys()))].dtype
    if str(dtype) in [str(dtype) for dtype in [mx.float16, mx.bfloat16, mx.float32]]:
        shards = make_shards(weights, max_file_size_gibibyte=5)
        shards_count = len(shards)
        shard_file_format = (
            "model-{:05d}-of-{:05d}.safetensors"
            if shards_count > 1
            else "model.safetensors"
        )

        for i, shard in enumerate(shards):
            shard_name = shard_file_format.format(i + 1, shards_count)
            mx.save_safetensors(str(save_dir / shard_name), shard)
    else:
        save_file(
            weights,
            os.path.join(save_dir, "model.safetensors"),
            metadata={"format": "pt"},
        )
