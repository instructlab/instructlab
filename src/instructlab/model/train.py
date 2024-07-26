# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
import logging
import os
import shutil

# Third Party
# pylint: disable=ungrouped-imports
import click

# First Party
from instructlab import clickext, utils
from instructlab.configuration import DEFAULTS, map_train_to_library

logger = logging.getLogger(__name__)


ADDITIONAL_ARGUMENTS = "additional_args"


@click.command()
@click.option(
    "--data-path",
    type=click.Path(file_okay=True),
    cls=clickext.ConfigOption,
    required=True,  # default from config
    help="Base directory where data is stored.",
    default=lambda: DEFAULTS.DATASETS_DIR,
)
@click.option(
    "--ckpt-output-dir",
    type=click.Path(),
    cls=clickext.ConfigOption,
    required=True,  # default from config
    default=lambda: DEFAULTS.CHECKPOINTS_DIR,
    help="output directory to store checkpoints in during training",
)
@click.option(
    "--data-output-dir",
    type=click.Path(),
    cls=clickext.ConfigOption,
    required=True,  # default from config
    default=lambda: DEFAULTS.INTERNAL_DIR,
    help="output directory to store training data in",
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
    help="HuggingFace model repo path, in the format of <namespace>/<repo_name>.",
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
    help="The number of times the training data is passed through the training algorithm. Please note that this value is used on Linux platforms only.",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda", "hpu"]),
    show_default=True,
    default="cpu",
    help=(
        "PyTorch device for Linux training. Use 'cuda' "
        "for NVidia CUDA / AMD ROCm GPU, to use specific GPU, set visible GPU before run train command."
    ),
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
    help="maximum length, in tokens, of a single sample.",
)
@click.option(
    "--max-batch-len",
    type=int,
    cls=clickext.ConfigOption,
    required=True,  # default from config
    help="maximum overall length of samples processed in a given batch.",
)
@click.option(
    "--effective-batch-size",
    type=int,
    cls=clickext.ConfigOption,
    required=True,  # default from config
    help="total batch size across all GPUs",
)
@click.option(
    "--save-samples",
    type=int,
    cls=clickext.ConfigOption,
    required=True,  # default from config
    help="The number of samples processed in between checkpoints.",
)
@click.option(
    "--learning-rate",
    type=float,
    cls=clickext.ConfigOption,
    config_sections=ADDITIONAL_ARGUMENTS,
    required=True,  # default from config
    help="learning rate for training",
)
@click.option(
    "--warmup-steps",
    type=int,
    cls=clickext.ConfigOption,
    config_sections=ADDITIONAL_ARGUMENTS,
    required=True,  # default from config
    help="warmup steps for training",
)
@click.option(
    "--deepspeed-cpu-offload-optimizer",
    type=bool,
    cls=clickext.ConfigOption,
    required=True,  # default from config
    # config_section="deepspeed_options",
    help="if true enables optimizer offload",
)
@click.option(
    "--deepspeed-cpu-offload-optimizer-ratio",
    type=float,
    cls=clickext.ConfigOption,
    required=True,  # default from config
    config_sections=ADDITIONAL_ARGUMENTS,
    help="cpu offload optimizer ratio",
)
@click.option(
    "--deepspeed-cpu-offload-optimizer-pin-memory",
    type=bool,
    cls=clickext.ConfigOption,
    required=True,  # default from config
    config_sections=ADDITIONAL_ARGUMENTS,
    help="if true pin memory when using cpu optimizer",
)
# below flags are invalid if lora == false
@click.option(
    "--lora-rank",
    type=int,
    cls=clickext.ConfigOption,
    required=True,  # default from config
    # config_section="lora",
    help="rank of update matricies",
)
@click.option(
    "--lora-alpha",
    type=int,
    cls=clickext.ConfigOption,
    required=True,  # default from config
    config_sections=ADDITIONAL_ARGUMENTS,
    help="how influential/strong lora tune will be",
)
@click.option(
    "--lora-dropout",
    type=float,
    cls=clickext.ConfigOption,
    required=True,  # default from config
    config_sections=ADDITIONAL_ARGUMENTS,
    help="dropout for LoRA layers",
)
@click.option(
    "--lora-target-modules",
    cls=clickext.ConfigOption,
    config_sections=ADDITIONAL_ARGUMENTS,
    multiple=True,
    default=[],
    help="LoRA modules to use",
)
@click.option(
    "--lora-quantize-dtype",
    type=str,
    default=None,
    help="quantization data type to use when training a LoRA.",
)
@click.option(
    "--is-padding-free",
    cls=clickext.ConfigOption,
    type=bool,
    help="whether or not we are training a padding free transformer.",
)
@click.option(
    "--gpus",
    "nproc_per_node",
    cls=clickext.ConfigOption,
    type=int,
    help="this is the number of GPUs to use. This is a torch specific arg and must be called nproc-per-node",
)
@click.option(
    "--nnodes",
    type=int,
    cls=clickext.ConfigOption,
    config_sections=ADDITIONAL_ARGUMENTS,
    required=True,  # default from config
    help="number of machines in the training pool.",
)
@click.option(
    "--node-rank",
    type=int,
    cls=clickext.ConfigOption,
    config_sections=ADDITIONAL_ARGUMENTS,
    required=True,  # default from config
    help="the rank of this machine in the training group.",
)
@click.option(
    "--rdzv-id",
    type=int,
    cls=clickext.ConfigOption,
    config_sections=ADDITIONAL_ARGUMENTS,
    required=True,  # default from config
    help="this is the training group ID. So, if there are multiple matching endpoints, only the machines with matching IDs can connect.",
)
@click.option(
    "--rdzv-endpoint",
    type=str,
    cls=clickext.ConfigOption,
    config_sections=ADDITIONAL_ARGUMENTS,
    required=True,  # default from config
    help="this is the rendezvous endpoint which other torchrun jobs will join on.",
)
@click.option(
    "--legacy",
    is_flag=True,
    default=False,
    help="if true, enables the legacy linux training codepath from release 0.17.0 and prior.",
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
    legacy,
    **kwargs,
):
    """
    Takes synthetic data generated locally with `ilab data generate` and the previous model and learns a new model using the MLX API.
    On success, writes newly learned model to {model_dir}/mlx_model, which is where `chatmlx` will look for a model.
    """
    if not input_dir:
        # By default, generate output-dir is used as train input-dir
        input_dir = ctx.obj.config.generate.output_dir

    if four_bit_quant and device != "cuda":
        ctx.fail("'--4-bit-quant' option requires '--device=cuda'")

    effective_data_dir = Path(data_path if data_path else DEFAULTS.DATASETS_DIR)
    train_file = Path(effective_data_dir / "train_gen.jsonl")
    test_file = Path(effective_data_dir / "test_gen.jsonl")
    ckpt_output_dir = Path(kwargs["ckpt_output_dir"])

    # NOTE: If given a data_dir, input-dir is ignored in favor of existing!
    if not data_path or data_path.strip() == DEFAULTS.DATASETS_DIR:
        data_path = str(effective_data_dir)
        if not os.path.exists(input_dir):
            click.secho(
                f"Could not read directory: {input_dir}",
                fg="red",
            )
            raise click.exceptions.Exit(1)

        try:
            os.makedirs(data_path, exist_ok=True)
        except OSError as exc:
            click.secho(
                f"Could not create data dir: {exc}",
                fg="red",
            )
            raise click.exceptions.Exit(1)

        # generated input files reverse sorted by modification time
        def get_files(directory: str, pattern: str) -> list[str]:
            return sorted(
                [str(p) for p in Path(directory).glob(pattern)],
                key=os.path.getmtime,
                reverse=True,
            )

        # ignore the test_file and train_file to prevent it from being copied back onto itself
        # see: https://github.com/instructlab/instructlab/pull/1685
        test_files = [
            f
            for f in get_files(input_dir, "test_*")
            if os.path.basename(f) != os.path.basename(test_file)
        ]
        train_files = [
            f
            for f in get_files(input_dir, "train_*")
            if os.path.basename(f) != os.path.basename(train_file)
        ]

        if not train_files or not test_files:
            click.secho(
                f"{input_dir} does not contain training or test files, did you run `ilab data generate`?",
                fg="red",
            )
            raise click.exceptions.Exit(1)
        if len(train_files) > 1 or len(test_files) > 1:
            click.secho(
                "Found multiple files from `ilab data generate`. Using the most recent generation.",
                fg="yellow",
            )
        # The first file is latest
        logger.debug("train_file=%s", train_files[0])
        logger.debug("test_file=%s", test_files[0])
        shutil.copy(train_files[0], train_file)
        shutil.copy(test_files[0], test_file)

    # if macos, preserve that path
    if utils.is_macos_with_m_chip():
        # Local
        from ..mlx_explore.gguf_convert_to_mlx import load
        from ..mlx_explore.utils import fetch_tokenizer_from_hub
        from ..train.lora_mlx.convert import convert_between_mlx_and_pytorch
        from ..train.lora_mlx.lora import load_and_train
        from ..train.lora_mlx.make_data import make_data

        if not skip_preprocessing:
            try:
                make_data(data_dir=data_path)
            except FileNotFoundError as exc:
                click.secho(
                    f"Could not read from data directory: {exc}",
                    fg="red",
                )
                raise click.exceptions.Exit(1)

        # NOTE we can skip this if we have a way ship MLX
        # PyTorch safetensors to MLX safetensors
        model_dir_local = model_path.replace("/", "-")
        model_dir_local = f"{ckpt_output_dir}/{model_dir_local}"
        model_dir_mlx = f"{model_dir_local}-mlx"
        model_dir_mlx_quantized = f"{model_dir_local}-mlx-q"

        if skip_quantize:
            dest_model_dir = model_dir_mlx
            quantize_arg = False
        else:
            dest_model_dir = model_dir_mlx_quantized
            quantize_arg = True

        if tokenizer_dir is not None and gguf_model_path is not None:
            if not local:
                tokenizer_dir_local = tokenizer_dir.replace("/", "-")
                fetch_tokenizer_from_hub(tokenizer_dir, tokenizer_dir_local)

            # no need to pass quantize_arg for now, script automatically detects if quantization is necessary based on whether gguf model is quantized or not
            load(gguf=gguf_model_path, repo=tokenizer_dir, mlx_path=dest_model_dir)

            for filename in os.listdir(model_dir_local):
                shutil.copy(
                    os.path.join(model_dir_local, filename),
                    os.path.join(dest_model_dir, filename),
                )
            shutil.rmtree(model_dir_local, ignore_errors=True)

        else:
            # Downloading PyTorch SafeTensor and Converting to MLX SafeTensor
            convert_between_mlx_and_pytorch(
                hf_path=model_path,
                mlx_path=dest_model_dir,
                quantize=quantize_arg,
                local=local,
            )

        adapter_file_path = f"{dest_model_dir}/adapters.npz"

        # train the model with LoRA
        load_and_train(
            model=dest_model_dir,
            train=True,
            data=data_path,
            adapter_file=adapter_file_path,
            iters=iters,
            save_every=10,
            steps_per_eval=10,
        )
    elif legacy:
        # Local
        from ..llamacpp.llamacpp_convert_to_gguf import convert_llama_to_gguf
        from ..train.linux_train import linux_train

        training_results_dir = linux_train(
            ctx=ctx,
            train_file=train_file,
            test_file=test_file,
            model_name=model_path,
            num_epochs=num_epochs,
            train_device=device,
            four_bit_quant=four_bit_quant,
        )

        final_results_dir = training_results_dir / "final"
        if final_results_dir.exists():
            shutil.rmtree(final_results_dir)
        final_results_dir.mkdir()

        gguf_models_dir = Path(DEFAULTS.CHECKPOINTS_DIR)
        gguf_models_dir.mkdir(exist_ok=True)
        gguf_models_file = gguf_models_dir / "ggml-model-f16.gguf"

        # Remove previously trained model, its taking up space we may need in the next step
        gguf_models_file.unlink(missing_ok=True)

        # TODO: Figure out what to do when there are multiple checkpoint dirs.
        # Right now it's just copying files from the first one numerically not necessarily the best one
        for fpath in (
            "checkpoint-*/added_tokens.json",
            "checkpoint-*/special_tokens_map.json",
            "checkpoint-*/tokenizer.json",
            "checkpoint-*/tokenizer.model",
            "checkpoint-*/tokenizer_config.json",
            "merged_model/config.json",
            "merged_model/generation_config.json",
        ):
            file_ = next(training_results_dir.glob(fpath))
            shutil.copy(file_, final_results_dir)
            print(f"Copied {file_} to {final_results_dir}")

        for file in training_results_dir.glob("merged_model/*.safetensors"):
            shutil.move(file, final_results_dir)
            print(f"Moved {file} to {final_results_dir}")

        if four_bit_quant:
            print(
                "SKIPPING CONVERSION to gguf. This is unsupported with --4-bit-quant. "
                + "See https://github.com/instructlab/instructlab/issues/579."
            )
            return

        gguf_file_path = convert_llama_to_gguf(model=final_results_dir, pad_vocab=True)

        # Remove safetensors files to save space, were done with them here
        # and the huggingface lib has them cached
        for file in final_results_dir.glob("*.safetensors"):
            file.unlink()

        shutil.move(gguf_file_path, gguf_models_file)
        print(f"Save trained model to {gguf_models_file}")

        # cleanup checkpoint dir since it's name is unpredictable
        # TODO: figure out how checkpoint dirs should be cleaned up
        # checkpoint_dirs = training_results_dir.glob("checkpoint*")
        # shutil.rmtree(checkpoint_dirs[0])
    else:
        # run_training is a dynamic attribute, pylint is not clever enough
        # to detect it.
        # Third Party
        from instructlab.training import (  # pylint: disable=no-name-in-module
            run_training,
        )

        # pull the trainrandom.randinting and torch args from the flags
        # the flags are populated from the config as a base.
        params = ctx.params

        if os.path.isdir(data_path):
            click.secho(
                f"Data path cannot be a directory. It must be a path to a valid jsonl file. Value given: {data_path}",
                fg="red",
            )
            raise click.exceptions.Exit(1)
        train_args, torch_args = map_train_to_library(params)
        try:
            run_training(train_args=train_args, torch_args=torch_args)
        # unsure what types of exceptions training library returns, this will catch all for now
        except Exception as exc:
            click.secho(
                f"Error while executing training library {exc}",
                fg="red",
            )
            raise click.exceptions.Exit(1)
