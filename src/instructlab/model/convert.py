# SPDX-License-Identifier: Apache-2.0

# Standard
from glob import glob
from pathlib import Path
import logging
import os
import shutil

# Third Party
from huggingface_hub import errors as hf_errors
from requests import exceptions as requests_exceptions
import click

# First Party
from instructlab import clickext, utils

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--model-dir",
    help="Base directory where model is stored.",
    show_default=True,
    required=True,
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
@click.pass_context
@clickext.display_params
@utils.macos_requirement(echo_func=click.secho, exit_exception=click.exceptions.Exit)
def convert(
    ctx,  # pylint: disable=unused-argument
    model_dir,
    adapter_file,
    skip_de_quantize,
    skip_quantize,
    model_name,
):
    """Converts model to GGUF"""
    # pylint: disable=import-outside-toplevel
    # Third Party
    from instructlab_quantize import run_quantize  # pylint: disable=import-error

    # Local
    from ..llamacpp.llamacpp_convert_to_gguf import convert_llama_to_gguf
    from ..train.lora_mlx.convert import convert_between_mlx_and_pytorch
    from ..train.lora_mlx.fuse import fine_tune

    model_dir = os.path.expandvars(os.path.expanduser(model_dir))

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
    try:
        fine_tune(
            model=source_model_dir,
            save_path=model_dir_fused,
            adapter_file=adapter_file,
            de_quantize=not skip_de_quantize,
        )
    except (requests_exceptions.HTTPError, hf_errors.HFValidationError) as e:
        click.secho(
            f"Failed to fine tune: {e}",
            fg="red",
        )
        raise click.exceptions.Exit(1)

    logger.info(f"deleting {source_model_dir}...")
    shutil.rmtree(source_model_dir)

    model_dir_fused_pt = f"{model_name}-trained"
    # this converts MLX to PyTorch
    convert_between_mlx_and_pytorch(
        hf_path=model_dir_fused, mlx_path=model_dir_fused_pt, local=True, to_pt=True
    )

    logger.info(f"deleting {model_dir_fused}...")
    shutil.rmtree(model_dir_fused)

    convert_llama_to_gguf(
        model=Path(model_dir_fused_pt),
        pad_vocab=True,
        skip_unknown=True,
        outfile=f"{model_dir_fused_pt}/{model_name}.gguf",
    )

    logger.info(f"deleting safetensors files from {model_dir_fused_pt}...")
    for file in glob(os.path.join(model_dir_fused_pt, "*.safetensors")):
        os.remove(file)

    # quantize to 4-bit GGUF (optional)
    if not skip_quantize:
        gguf_model_dir = f"{model_dir_fused_pt}/{model_name}.gguf"
        gguf_model_q_dir = f"{model_dir_fused_pt}/{model_name}-Q4_K_M.gguf"
        run_quantize(gguf_model_dir, gguf_model_q_dir, "Q4_K_M")

    logger.info(f"deleting {model_dir_fused_pt}/{model_name}.gguf...")
    os.remove(os.path.join(model_dir_fused_pt, f"{model_name}.gguf"))
