# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
import glob
import os
import shutil

# Third Party
import click

# First Party
from instructlab import utils


@click.command()
@click.option(
    "--model-dir",
    help="Base directory where model is stored.",
    default="instructlab-merlinite-7b-lab-mlx-q",
    show_default=True,
)
@click.option("--adapter-file", help="LoRA adapter to fuse.", default=None)
@click.option(
    "-sd",
    "--skip-de-quantize",
    help="Skip de-quantization.",
    is_flag=True,
)
@click.option(
    "-sq",
    "--skip-quantize",
    is_flag=True,
    help="Whether to skip quantization while converting to GGUF.",
)
@click.option(
    "--model-name",
    help="Name of the model being trained/converted. Informs the naming of the final trained model file",
    default=None,
    show_default=True,
)
@utils.macos_requirement(echo_func=click.secho, exit_exception=click.exceptions.Exit)
@click.pass_context
@utils.display_params
def convert(ctx, model_dir, adapter_file, skip_de_quantize, skip_quantize, model_name):
    """Converts model to GGUF"""
    # pylint: disable=C0415
    # Third Party
    from instructlab_quantize import run_quantize  # pylint: disable=import-error

    # Local
    from ..llamacpp.llamacpp_convert_to_gguf import convert_llama_to_gguf
    from ..train.lora_mlx.convert import convert_between_mlx_and_pytorch
    from ..train.lora_mlx.fuse import fine_tune

    # compute model name from model-dir if not supplied
    if model_name is None:
        mlx_q_suffix = "-mlx-q"
        model_name = model_dir.split("/")[-1]
        model_name = model_name.replace(mlx_q_suffix, "")

    if adapter_file is None:
        adapter_file = os.path.join(model_dir, "adapters.npz")
    source_model_dir = model_dir
    model_dir_fused = f"{source_model_dir}-fused"

    # this combines adapter with the original model to produce the updated model
    fine_tune(
        model=source_model_dir,
        save_path=model_dir_fused,
        adapter_file=adapter_file,
        de_quantize=not skip_de_quantize,
    )

    ctx.obj.logger.info(f"deleting {source_model_dir}...")
    shutil.rmtree(source_model_dir)

    model_dir_fused_pt = f"{model_name}-trained"
    # this converts MLX to PyTorch
    convert_between_mlx_and_pytorch(
        hf_path=model_dir_fused, mlx_path=model_dir_fused_pt, local=True, to_pt=True
    )

    ctx.obj.logger.info(f"deleting {model_dir_fused}...")
    shutil.rmtree(model_dir_fused)

    convert_llama_to_gguf(
        model=Path(model_dir_fused_pt),
        pad_vocab=True,
        skip_unknown=True,
        outfile=f"{model_dir_fused_pt}/{model_name}.gguf",
    )

    ctx.obj.logger.info(f"deleting safetensors files from {model_dir_fused_pt}...")
    for file in glob(os.path.join(model_dir_fused_pt, "*.safetensors")):
        os.remove(file)

    # quantize to 4-bit GGUF (optional)
    if not skip_quantize:
        gguf_model_dir = f"{model_dir_fused_pt}/{model_name}.gguf"
        gguf_model_q_dir = f"{model_dir_fused_pt}/{model_name}-Q4_K_M.gguf"
        run_quantize(gguf_model_dir, gguf_model_q_dir, "Q4_K_M")

    ctx.obj.logger.info(f"deleting {model_dir_fused_pt}/{model_name}.gguf...")
    os.remove(os.path.join(model_dir_fused_pt, f"{model_name}.gguf"))
