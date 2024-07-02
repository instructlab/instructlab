# SPDX-License-Identifier: MIT
# Copyright © 2023 Apple Inc.

# Standard
from pathlib import Path
from typing import Any, Generator
import glob
import json
import logging
import os

# Third Party
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
import mlx.core as mx
import mlx.nn as nn
import transformers

# Local
from .models import llama, mixtral, phi2

# Constants
MODEL_MAPPING = {
    "llama": llama,
    "mistral": llama,  # mistral is compatible with llama
    "phi": phi2,
    "mixtral": mixtral,
}


def _get_classes(config: dict):
    """
    Retrieve the model and model args classes based on the configuration.

    Args:
        config (dict): The model configuration.

    Returns:
        A tuple containing the Model class and the ModelArgs class.
    """
    model_type = config["model_type"]
    if model_type not in MODEL_MAPPING:
        msg = f"Model type {model_type} not supported."
        logging.error(msg)
        raise ValueError(msg)
    print(f"Using {model_type=}")
    arch = MODEL_MAPPING[model_type]
    return arch.Model, arch.ModelArgs


def fetch_from_hub(hf_path: str, local: bool):
    if local:
        model_path = hf_path
    else:
        model_path = snapshot_download(
            repo_id=hf_path,
            local_dir=hf_path.replace(
                "/", "-"
            ),  # "instructlab/merlinite-7b-lab" to "instructlab-merlinite-7b-lab"
            allow_patterns=["*.json", "*.safetensors", "tokenizer.model"],
        )

    weight_files = glob.glob(f"{model_path}/*.safetensors")
    if len(weight_files) == 0:
        raise FileNotFoundError("No safetensors found in {}".format(model_path))

    weights = {}
    for wf in weight_files:
        w = mx.load(wf, return_metadata=False)
        weights.update(w.items())

    config = transformers.AutoConfig.from_pretrained(hf_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        hf_path,
    )
    return weights, config.to_dict(), tokenizer


def upload_to_hub(path: str, name: str, hf_path: str):
    # Standard
    import os

    # Third Party
    from huggingface_hub import HfApi, ModelCard, logging

    repo_id = f"mlx-community/{name}"

    card = ModelCard.load(hf_path)
    card.data.tags = ["mlx"] if card.data.tags is None else card.data.tags + ["mlx"]
    card.text = f"""
# {name}
This model was converted to MLX format from [`{hf_path}`]().
Refer to the [original model card](https://huggingface.co/{hf_path}) for more details on the model.
## Use with mlx
```bash
pip install mlx
git clone https://github.com/ml-explore/mlx-examples.git
cd mlx-examples/llms/hf_llm
python generate.py --model {repo_id} --prompt "My name is"
```
"""
    card.save(os.path.join(path, "README.md"))

    logging.set_verbosity_info()

    api = HfApi()
    api.create_repo(repo_id=repo_id, exist_ok=True)
    api.upload_folder(
        folder_path=path,
        repo_id=repo_id,
        repo_type="model",
    )


def make_shards(weights: dict, max_file_size_gibibyte: int = 15):
    max_file_size_bytes = max_file_size_gibibyte << 30
    shards = []
    shard: dict[str, Any] = {}
    shard_size = 0
    for k, v in weights.items():
        if shard_size + v.nbytes > max_file_size_bytes:
            shards.append(shard)
            shard, shard_size = {}, 0
        shard[k] = v
        shard_size += v.nbytes
    shards.append(shard)
    return shards


def save_model(save_dir: str, weights, tokenizer, config):
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)

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
            mx.save_safetensors(str(save_dir_path / shard_name), shard)
    else:
        save_file(
            weights,
            os.path.join(save_dir_path, "model.safetensors"),
            metadata={"format": "pt"},
        )

    tokenizer.save_pretrained(save_dir_path)
    tokenizer.save_vocabulary(save_dir_path)

    with open(save_dir_path / "config.json", "w") as fid:
        json.dump(config, fid, indent=4)


def load(path_or_hf_repo: str):
    # If the path exists, it will try to load model form it
    # otherwise download and cache from the hf_repo and cache
    model_path = Path(os.path.expandvars(os.path.expanduser(path_or_hf_repo)))
    if not model_path.exists():
        model_path = Path(
            snapshot_download(
                repo_id=path_or_hf_repo,
                allow_patterns=["*.json", "*.safetensors", "tokenizer.model"],
            )
        )

    with open(model_path / "config.json", "r") as f:
        config = json.loads(f.read())
        quantization = config.get("quantization", None)

    weight_files = glob.glob(str(model_path / "*.safetensors"))
    if len(weight_files) == 0:
        raise FileNotFoundError("No safetensors found in {}".format(model_path))

    weights = {}
    for wf in weight_files:
        w = mx.load(wf, return_metadata=False)
        weights.update(w.items())

    model_class, model_args_class = _get_classes(config=config)
    model_args = model_args_class.from_dict(config)
    model = model_class(model_args)
    if quantization is not None:
        nn.QuantizedLinear.quantize_module(
            model,
            **quantization,
            linear_class_predicate=lambda m: isinstance(m, nn.Linear)
            and m.weight.shape[0] != 8,
        )

    model.load_weights(list(weights.items()))

    mx.eval(model.parameters())
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer, config


def generate(
    prompt: mx.array, model: nn.Module, temp: float = 0.0
) -> Generator[mx.array, None, None]:
    """
    Generate text based on the given prompt and model.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.
        temp (float): The temperature for sampling. If temp is 0, use max sampling.

    Yields:
        mx.array: The generated text.
    """

    def sample(logits: mx.array) -> mx.array:
        return (
            mx.argmax(logits, axis=-1)
            if temp == 0
            else mx.random.categorical(logits * (1 / temp))
        )

    y = prompt
    cache = None
    while True:
        logits, cache = model(y[None], cache=cache)
        logits = logits[:, -1, :]
        y = sample(logits)
        yield y
