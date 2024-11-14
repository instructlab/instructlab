# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
import enum
import logging
import os
import pathlib

# Third Party
# pylint: disable=ungrouped-imports
import click

# First Party
from instructlab import clickext
from instructlab.configuration import DEFAULTS, map_train_to_library

logger = logging.getLogger(__name__)

ADDITIONAL_ARGUMENTS = "additional_args"


class SupportedTrainingStrategies(enum.Enum):
    """Available advanced training strategies"""

    LAB_MULTIPHASE: str = "lab-multiphase"


def clickpath_setup(is_dir: bool) -> click.Path:
    """
    Creates a click.Path object meeting requirements:
        - path to target must exist,
        - file vs. directory are mutually exclusive,
        - path is fully resolved.
    """

    return click.Path(
        exists=True,
        file_okay=not is_dir,
        dir_okay=is_dir,
        resolve_path=True,
        path_type=pathlib.Path,
    )


@click.command()
@click.option(
    "--data-path",
    type=click.Path(file_okay=True),
    cls=clickext.ConfigOption,
    required=True,  # default from config
)
@click.option(
    "--ckpt-output-dir",
    type=click.Path(),
    cls=clickext.ConfigOption,
    required=True,  # default from config
)
@click.option(
    "--data-output-dir",
    type=click.Path(),
    cls=clickext.ConfigOption,
    required=True,  # default from config
)
@click.option(
    "--input-dir",
    type=click.Path(),
    show_default=True,  # TODO: set to None and change help message
    help="Path to generated files to use as input.",
)
@click.option(
    "--gguf-model-path",
    help="Local directory where gguf model is stored.",
    default=None,
    show_default=True,
)
@click.option(
    "--skip-preprocessing",
    help="Skips data preprocessing step for MLX training if data is already cached.",
    is_flag=True,
)
@click.option(
    "--tokenizer-dir",
    type=click.Path(),
    help="Base directory where tokenizer is stored.",
    default=None,
    show_default=True,
)
@click.option(
    "--model-path",
    type=click.Path(),
    cls=clickext.ConfigOption,
    required=True,  # default from config
    default=DEFAULTS.MODEL_REPO,
)
@click.option(
    "--iters",
    help="Number of iterations to train LoRA.",
    default=100,
)
@click.option(
    "--local",
    is_flag=True,
    help="Whether or not `model_dir` is remote from HuggingFace.",
)
@click.option(
    "-sq",
    "--skip-quantize",
    is_flag=True,
    help="Whether to skip quantization while converting to MLX. This parameter will be ignored if --gguf-model-path and --tokenizer-dir are specified.",
)
@click.option(
    "--num-epochs",
    type=click.INT,
    cls=clickext.ConfigOption,
    required=True,  # default from config
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda", "hpu", "mps"]),
    cls=clickext.ConfigOption,
)
@click.option(
    "--4-bit-quant",
    "four_bit_quant",
    is_flag=True,
    show_default=True,
    default=False,
    # TODO: hidden option until llamacpp_convert_to_gguf.py supports
    # quantized models, https://github.com/instructlab/instructlab/issues/579
    hidden=True,
    help=(
        "Use BitsAndBytes for 4-bit quantization "
        "(reduces GPU VRAM usage and may slow down training)"
    ),
)
@click.option(
    "--max-seq-len",
    type=int,
    cls=clickext.ConfigOption,
    required=True,  # default from config
)
@click.option(
    "--max-batch-len",
    type=int,
    cls=clickext.ConfigOption,
    required=True,  # default from config
)
@click.option(
    "--effective-batch-size",
    type=int,
    cls=clickext.ConfigOption,
    required=True,  # default from config
)
@click.option(
    "--save-samples",
    type=int,
    cls=clickext.ConfigOption,
    required=True,  # default from config
)
@click.option(
    "--learning-rate",
    type=float,
    cls=clickext.ConfigOption,
    config_sections=ADDITIONAL_ARGUMENTS,
    required=True,  # default from config
)
@click.option(
    "--warmup-steps",
    type=int,
    cls=clickext.ConfigOption,
    config_sections=ADDITIONAL_ARGUMENTS,
    required=True,  # default from config
)
@click.option(
    "--deepspeed-cpu-offload-optimizer",
    type=bool,
    cls=clickext.ConfigOption,
    required=True,  # default from config
    # config_section="deepspeed_options",
)
@click.option(
    "--deepspeed-cpu-offload-optimizer-ratio",
    type=float,
    cls=clickext.ConfigOption,
    required=True,  # default from config
    config_sections=ADDITIONAL_ARGUMENTS,
)
@click.option(
    "--deepspeed-cpu-offload-optimizer-pin-memory",
    type=bool,
    cls=clickext.ConfigOption,
    required=True,  # default from config
    config_sections=ADDITIONAL_ARGUMENTS,
)
@click.option(
    "--fsdp-cpu-offload-optimizer",
    type=bool,
    cls=clickext.ConfigOption,
)
@click.option(
    "--distributed-backend",
    type=str,
    cls=clickext.ConfigOption,
)
# below flags are invalid if lora == false
@click.option(
    "--lora-rank",
    type=int,
    cls=clickext.ConfigOption,
    # config_section="lora",
)
@click.option(
    "--lora-alpha",
    type=int,
    cls=clickext.ConfigOption,
    config_sections=ADDITIONAL_ARGUMENTS,
)
@click.option(
    "--lora-dropout",
    type=float,
    cls=clickext.ConfigOption,
    config_sections=ADDITIONAL_ARGUMENTS,
)
@click.option(
    "--lora-target-modules",
    cls=clickext.ConfigOption,
    config_sections=ADDITIONAL_ARGUMENTS,
    multiple=True,
    default=[],
)
@click.option(
    "--lora-quantize-dtype",
    type=str,
    cls=clickext.ConfigOption,
    default=None,
)
@click.option(
    "--is-padding-free",
    cls=clickext.ConfigOption,
    type=bool,
)
@click.option(
    "--use-dolomite",
    cls=clickext.ConfigOption,
    config_sections=ADDITIONAL_ARGUMENTS,
    type=click.BOOL,
    required=True,  # default from config
)
@click.option(
    "--gpus",
    "nproc_per_node",
    cls=clickext.ConfigOption,
    type=int,
)
@click.option(
    "--nnodes",
    type=int,
    cls=clickext.ConfigOption,
    config_sections=ADDITIONAL_ARGUMENTS,
    required=True,  # default from config
)
@click.option(
    "--node-rank",
    type=int,
    cls=clickext.ConfigOption,
    config_sections=ADDITIONAL_ARGUMENTS,
    required=True,  # default from config
)
@click.option(
    "--rdzv-id",
    type=int,
    cls=clickext.ConfigOption,
    config_sections=ADDITIONAL_ARGUMENTS,
    required=True,  # default from config
)
@click.option(
    "--rdzv-endpoint",
    type=str,
    cls=clickext.ConfigOption,
    config_sections=ADDITIONAL_ARGUMENTS,
    required=True,  # default from config
)
@click.option("--disable-flash-attn", is_flag=True, cls=clickext.ConfigOption)
@click.option(
    "--strategy",
    type=click.Choice(
        [SupportedTrainingStrategies.LAB_MULTIPHASE.value], case_sensitive=False
    ),
    show_default=True,
    help="If chosen, will run the selected training strategy instead of a single training run.",
)
@click.option(
    "--phased-base-dir",
    type=clickpath_setup(is_dir=True),
    cls=clickext.ConfigOption,
)
@click.option(
    "--phased-phase1-data",
    type=clickpath_setup(is_dir=False),
    help="Path to .jsonl file that will be used for the first phase of end-to-end training.",
)
@click.option(
    "--phased-phase1-num-epochs",
    cls=clickext.ConfigOption,
    type=click.IntRange(min=1),
)
@click.option(
    "--phased-phase1-samples-per-save",
    cls=clickext.ConfigOption,
    type=click.IntRange(min=0),
)
@click.option(
    "--phased-phase1-effective-batch-size",
    cls=clickext.ConfigOption,
    type=click.IntRange(min=1),
)
@click.option(
    "--phased-phase2-data",
    type=clickpath_setup(is_dir=False),
    help="Path to .jsonl file that will be used for the second phase of end-to-end training.",
)
@click.option(
    "--phased-phase2-num-epochs",
    cls=clickext.ConfigOption,
    type=click.IntRange(min=1),
)
@click.option(
    "--phased-phase2-samples-per-save",
    cls=clickext.ConfigOption,
    type=click.IntRange(min=0),
)
@click.option(
    "--phased-phase2-effective-batch-size",
    cls=clickext.ConfigOption,
    type=click.IntRange(min=1),
)
@click.option(
    "--phased-mt-bench-judge",
    # type=clickpath_setup(is_dir=True), # want this in the future, can't guarantee it exists so can't enforce it this way.
    type=click.Path(dir_okay=True, file_okay=False, path_type=pathlib.Path),
    cls=clickext.ConfigOption,
)
@click.option(
    "--skip-user-confirm",
    "-y",
    is_flag=True,
    help="Skips any user confirmation prompts.",
)
@click.option(
    "--enable-serving-output",
    is_flag=True,
    help="Print serving engine logs during phased training checkpoint evaluation.",
)
@click.option(
    "--checkpoint-at-epoch",
    is_flag=True,
    help="By default, checkpoints are saved at the end of each training epoch. This option disables this behavior.",
)
@click.option(
    "--pipeline",
    type=click.Choice(["simple", "full", "accelerated"]),
    cls=clickext.ConfigOption,
)
@click.option(
    "--training-journal",
    cls=clickext.ConfigOption,
)
@click.option(
    "--force-clear-phased-cache",
    is_flag=True,
    help="Clear phased cache (journal, checkpoints, metadata). Helpful paired with '--skip-user-confirm'",
)
@click.option(
    "--optimize-memory",
    is_flag=True,
    help="Optimize Memory Usage on CPU and MacOS. This uses the torch_dtype='auto' instead of float32",
)
@click.pass_context
@clickext.display_params
def train(
    ctx,
    data_path: str,
    input_dir,
    skip_preprocessing,
    tokenizer_dir,
    gguf_model_path,
    model_path,
    iters,
    local,
    skip_quantize,
    num_epochs,
    device: str,
    four_bit_quant: bool,
    strategy: str | None,
    phased_base_dir: pathlib.Path,
    phased_phase1_data: pathlib.Path | None,
    phased_phase1_num_epochs: int | None,
    phased_phase1_samples_per_save: int | None,
    phased_phase1_effective_batch_size: int | None,
    phased_phase2_data: pathlib.Path | None,
    phased_phase2_num_epochs: int | None,
    phased_phase2_samples_per_save: int | None,
    phased_phase2_effective_batch_size: int | None,
    phased_mt_bench_judge: pathlib.Path | None,
    skip_user_confirm: bool,
    enable_serving_output: bool,
    pipeline: str,
    training_journal: pathlib.Path | None,
    force_clear_phased_cache: bool,
    distributed_backend,
    optimize_memory,
    **kwargs,
):
    """
    Takes synthetic data generated locally with `ilab data generate` and the previous model and learns a new model using the MLX API.
    On success, writes newly learned model to {model_dir}/mlx_model, which is where `chatmlx` will look for a model.
    """
    if (
        pipeline in ("full", "simple")
        and strategy == SupportedTrainingStrategies.LAB_MULTIPHASE.value
    ):
        ctx.fail("Multi Phase training is only supported with `--pipeline accelerated`")

    # TODO: cdoern, remove this flag
    if not input_dir:
        # By default, generate output-dir is used as train input-dir
        input_dir = ctx.obj.config.generate.output_dir

    if four_bit_quant and device != "cuda":
        ctx.fail("'--4-bit-quant' option requires '--device=cuda'")

    if (
        pipeline in ("full", "accelerated")
    ) and strategy != SupportedTrainingStrategies.LAB_MULTIPHASE.value:
        if not os.path.isfile(data_path):
            ctx.fail(
                f"Data path must be to a valid .jsonl file. Value given: {data_path}"
            )
    # we can use train_args locally to run lower fidelity training
    if is_high_fidelity(device=device) and pipeline == "accelerated":
        train_args, torch_args = map_train_to_library(ctx, ctx.params)

        # First Party
        from instructlab.model import accelerated_train

        try:
            accelerated_train.accelerated_train(
                train_args=train_args,
                torch_args=torch_args,
                strategy=strategy,
                distributed_backend=distributed_backend,
                phased_phase1_data=phased_phase1_data,
                phased_phase2_data=phased_phase2_data,
                phased_base_dir=phased_base_dir,
                phased_phase1_num_epochs=phased_phase1_num_epochs,
                phased_phase1_samples_per_save=phased_phase1_samples_per_save,
                phased_phase1_effective_batch_size=phased_phase1_effective_batch_size,
                phased_phase2_num_epochs=phased_phase2_num_epochs,
                phased_phase2_samples_per_save=phased_phase2_samples_per_save,
                phased_phase2_effective_batch_size=phased_phase2_effective_batch_size,
                enable_serving_output=enable_serving_output,
                phased_mt_bench_judge=phased_mt_bench_judge,
                skip_user_confirm=skip_user_confirm,
                force_clear_phased_cache=force_clear_phased_cache,
                eval_serve=ctx.obj.config.serve,
                eval_gpus=ctx.obj.config.evaluate.gpus,
                training_journal=training_journal,
            )
        except Exception as exc:
            click.secho(f"Accelerated Training failed with {str(exc)}")
            raise click.exceptions.Exit(1)
    elif not is_high_fidelity(device=device) and pipeline == "full":
        # Third Party
        import torch

        torch.set_autocast_enabled(False)
        # First Party
        from instructlab.model import full_train

        train_args, torch_args = map_train_to_library(ctx, ctx.params)
        # if on CPU or MPS, execute full train, which is based
        # off of the structure of the training repo, just with different optimizers, model sizes, and special data gradient accumulation to get it
        # to fit on most consumer laptops
        full_train.train(
            train_args=train_args, device=device, optimize_memory=optimize_memory
        )
    elif pipeline == "simple":
        # First Party
        from instructlab.model import simple_train

        try:
            simple_train.simple_train(
                model_path=model_path,
                skip_preprocessing=skip_preprocessing,
                skip_quantize=skip_quantize,
                gguf_model_path=gguf_model_path,
                tokenizer_dir=tokenizer_dir,
                data_path=data_path,
                input_dir=input_dir,
                ckpt_output_dir=Path(kwargs["ckpt_output_dir"]),
                iters=iters,
                local=local,
                num_epochs=num_epochs,
                device=device,
                four_bit_quant=four_bit_quant,
            )
        except Exception as exc:
            click.secho(f"{exc}", fg="red")
            raise click.exceptions.Exit(1)
    else:
        click.secho(
            f"Unable to train with device={device} and pipeline={pipeline}", fg="red"
        )
        raise click.exceptions.Exit(1)


# chooses which type of training to run depending on the device provided
def is_high_fidelity(device):
    return device in ("cuda", "hpu")
