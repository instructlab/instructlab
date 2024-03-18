# Copyright © 2023 Apple Inc.

# Standard
from pathlib import Path

# Third Party
from mlx.utils import tree_flatten, tree_unflatten
from models.lora import LoRALinear
import click
import mlx.core as mx
import mlx.nn as nn
import utils


@click.command()
@click.option(
    "--model",
    default="mlx_model",
    help="The path to the local model directory or Hugging Face repo.",
)
@click.option(
    "--save-path",
    default="lora_fused_model",
    help="The path to save the fused model.",
)
@click.option(
    "--adapter-file",
    type=click.STRING,
    default="adapters.npz",
    help="Path to the trained adapter weights (npz or safetensors).",
)
@click.option(
    "--hf-path",
    help=(
        "Path to the original Hugging Face model. This is "
        "required for upload if --model is a local directory."
    ),
    type=click.STRING,
    default=None,
)
@click.option(
    "--upload-name",
    help="The name of model to upload to Hugging Face MLX Community",
    type=click.STRING,
    default=None,
)
@click.option(
    "--de-quantize",
    "-d",
    help="Generate a de-quantized model.",
    is_flag=True,
)
def fine_tune(model, save_path, adapter_file, hf_path, upload_name, de_quantize):
    """LoRA or QLoRA fine tuning."""
    print("Loading pretrained model")

    loaded_model, tokenizer, config = utils.load(model)

    # Load adapters and get number of LoRA layers
    adapters = list(mx.load(adapter_file).items())
    lora_layers = len([m for m in adapters if "q_proj.lora_a" in m[0]])

    # Freeze all layers other than LORA linears
    loaded_model.freeze()
    for l in loaded_model.model.layers[len(loaded_model.model.layers) - lora_layers :]:
        l.self_attn.q_proj = LoRALinear.from_linear(l.self_attn.q_proj)
        l.self_attn.v_proj = LoRALinear.from_linear(l.self_attn.v_proj)
        if hasattr(l, "block_sparse_moe"):
            l.block_sparse_moe.gate = LoRALinear.from_linear(l.block_sparse_moe.gate)

    loaded_model.update(tree_unflatten(adapters))
    fused_linears = [
        (n, m.to_linear())
        for n, m in loaded_model.named_modules()
        if isinstance(m, LoRALinear)
    ]

    loaded_model.update_modules(tree_unflatten(fused_linears))

    if de_quantize:
        de_quantize_layers = []
        for n, m in loaded_model.named_modules():
            if isinstance(m, nn.QuantizedLinear):
                bias = "bias" in m
                weight = m.weight
                weight = mx.dequantize(
                    weight,
                    m.scales,
                    m.biases,
                    m.group_size,
                    m.bits,
                ).astype(mx.float16)
                output_dims, input_dims = weight.shape
                linear = nn.Linear(input_dims, output_dims, bias=bias)
                linear.weight = weight
                if bias:
                    linear.bias = m.bias
                de_quantize_layers.append((n, linear))

        loaded_model.update_modules(tree_unflatten(de_quantize_layers))

    weights = dict(tree_flatten(loaded_model.parameters()))
    if de_quantize:
        config.pop("quantization", None)
    utils.save_model(save_path, weights, tokenizer, config)

    if upload_name is not None:
        if not Path(model).exists():
            # If the model path doesn't exist, assume it's an HF repo
            hf_path = model
        elif hf_path is None:
            raise ValueError(
                "Must provide original Hugging Face repo to upload local model."
            )
        utils.upload_to_hub(save_path, upload_name, hf_path)


if __name__ == "__main__":
    fine_tune()
