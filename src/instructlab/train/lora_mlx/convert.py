# SPDX-License-Identifier: MIT
# Copyright © 2023 Apple Inc.

# Standard
from typing import Optional
import copy

# Third Party
from mlx.utils import tree_flatten
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch

# Local
from . import utils


def quantize_model(weights, config, q_group_size, q_bits):
    quantized_config = copy.deepcopy(config)

    # Get model classes
    model_class, model_args_class = utils._get_classes(config=config)

    # Load the model:
    model = model_class(model_args_class.from_dict(config))
    model.load_weights(list(weights.items()))

    # Quantize the model:
    nn.QuantizedLinear.quantize_module(
        model,
        q_group_size,
        q_bits,
        linear_class_predicate=lambda m: isinstance(m, nn.Linear)
        and m.weight.shape[0] != 8,
    )

    # Update the config:
    quantized_config["quantization"] = {
        "group_size": q_group_size,
        "bits": q_bits,
    }
    quantized_weights = dict(tree_flatten(model.parameters()))

    return quantized_weights, quantized_config


def convert_between_mlx_and_pytorch(
    hf_path: str,
    mlx_path: str = "mlx_model",
    quantize: bool = False,
    q_group_size: int = 64,
    q_bits: int = 4,
    dtype: str = "float16",
    upload_name: Optional[str] = None,
    to_pt: bool = False,
    local: bool = False,
):
    """Convert Hugging Face model to MLX format"""
    if dtype not in ("float16", "bfloat16", "float32"):
        raise  # TODO something

    print("[INFO] Loading")
    weights, config, tokenizer = utils.fetch_from_hub(hf_path, local)

    if to_pt:
        dtype = np.float16 if quantize else getattr(np, dtype)
        print(f"{dtype=}")
        weights = {
            k: torch.from_numpy(np.array(v, copy=False, dtype=dtype))
            for k, v in weights.items()
        }
    else:
        dtype = mx.float16 if quantize else getattr(mx, dtype)
        print(f"{dtype=}")
        weights = {k: v.astype(dtype) for k, v in weights.items()}
    if quantize:
        print("[INFO] Quantizing")
        weights, config = quantize_model(weights, config, q_group_size, q_bits)

    utils.save_model(mlx_path, weights, tokenizer, config)
    if upload_name is not None:
        utils.upload_to_hub(mlx_path, upload_name, hf_path)
