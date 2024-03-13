# Copyright © 2023 Apple Inc.

# Standard
import copy

# Third Party
from mlx.utils import tree_flatten
import click
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
import utils


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


@click.command()
@click.option("--hf-path", type=click.STRING, help="Path to the Hugging Face model.")
@click.option(
    "--mlx-path",
    type=click.STRING,
    default="mlx_model",
    help="Path to save the MLX model.",
)
@click.option("--quantize", "-q", help="Generate a quantized model.", is_flag=True)
@click.option(
    "--q-group-size", help="Group size for quantization.", type=click.INT, default=64
)
@click.option(
    "--q-bits", help="Bits per weight for quantization.", type=click.INT, default=4
)
@click.option(
    "--dtype",
    help="Type to save the parameters, ignored if -q is given.",
    type=click.Choice(["float16", "bfloat16", "float32"], case_sensitive=True),
    default="float16",
)
@click.option(
    "--upload-name",
    help="The name of model to upload to Hugging Face MLX Community",
    type=click.STRING,
    default=None,
)
@click.option("--to-pt", is_flag=True)
@click.option("--local", is_flag=True)
def convert(
    hf_path, mlx_path, quantize, q_group_size, q_bits, dtype, upload_name, to_pt, local
):
    """Convert Hugging Face model to MLX format"""
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


if __name__ == "__main__":
    convert()
