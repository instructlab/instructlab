# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
import enum
import functools
import logging
import multiprocessing
import os
import pathlib
import pprint
import shutil
import typing

# Third Party
from instructlab.training import TorchrunArgs, TrainingArgs

# pylint: disable=ungrouped-imports
import click

# First Party
from instructlab import clickext, utils
from instructlab.configuration import DEFAULTS, map_train_to_library
from instructlab.model.backends import backends

logger = logging.getLogger(__name__)

ADDITIONAL_ARGUMENTS = "additional_args"


class SupportedTrainingStrategies(enum.Enum):
    """Available advanced training stratefies"""

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
    help="Base directory where data is stored.",
)
@click.option(
    "--ckpt-output-dir",
    type=click.Path(),
    cls=clickext.ConfigOption,
    required=True,  # default from config
    help="output directory to store checkpoints in during training",
)
@click.option(
    "--data-output-dir",
    type=click.Path(),
    cls=clickext.ConfigOption,
    required=True,  # default from config
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
    # config_section="lora",
    help="rank of update matricies",
)
@click.option(
    "--lora-alpha",
    type=int,
    cls=clickext.ConfigOption,
    config_sections=ADDITIONAL_ARGUMENTS,
    help="how influential/strong lora tune will be",
)
@click.option(
    "--lora-dropout",
    type=float,
    cls=clickext.ConfigOption,
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
    cls=clickext.ConfigOption,
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
    help="if true, enables the legacy linux training code path from release 0.17.0 and prior.",
)
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
    help="Base directory for organization of end-to-end intermediate outputs.",
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
    help="Number of epochs to run for first phase of end-to-end training.",
)
@click.option(
    "--phased-phase1-samples-per-save",
    cls=clickext.ConfigOption,
    type=click.IntRange(min=0),
    help="Number of samples to train on between saves for first phase of end-to-end training.",
)
@click.option(
    "--phased-phase1-effective-batch-size",
    cls=clickext.ConfigOption,
    type=click.IntRange(min=1),
    help="Total size of a training batch over all GPUs for Phase 1",
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
    help="Number of epochs to run for second phase of end-to-end training.",
)
@click.option(
    "--phased-phase2-samples-per-save",
    cls=clickext.ConfigOption,
    type=click.IntRange(min=0),
    help="Number of samples to train on between saves for second phase of end-to-end training.",
)
@click.option(
    "--phased-phase2-effective-batch-size",
    cls=clickext.ConfigOption,
    type=click.IntRange(min=1),
    help="Total size of a training batch over all GPUs for Phase 2",
)
@click.option(
    "--phased-mt-bench-judge",
    # type=clickpath_setup(is_dir=True), # want this in the future, can't guarantee it exists so can't enforce it this way.
    type=click.Path(dir_okay=True, file_okay=False, path_type=pathlib.Path),
    cls=clickext.ConfigOption,
    help="MT-Bench judge. Should be absolute path to local judge model directory. Download with `ilab model download` if necessary.",
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

    effective_data_dir: pathlib.Path = Path(
        data_path if data_path else DEFAULTS.DATASETS_DIR
    )
    train_file = effective_data_dir / "train_gen.jsonl"
    test_file = effective_data_dir / "test_gen.jsonl"
    ckpt_output_dir = Path(kwargs["ckpt_output_dir"])

    # NOTE: If given a data_dir, input-dir is ignored in favor of existing!
    if not data_path or data_path.strip() == DEFAULTS.DATASETS_DIR and not strategy:
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
    if utils.is_macos_with_m_chip() and not strategy:
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
        train_args, torch_args = map_train_to_library(ctx, ctx.params)
        logger.debug(
            "Rendered training arguments:\n%s", pprint.pformat(train_args.model_dump())
        )

        if strategy == SupportedTrainingStrategies.LAB_MULTIPHASE.value:
            if not (phased_phase1_data and phased_phase2_data):
                raise ctx.fail(
                    "End-to-end training minimally requires: `--phased-phase1-data`, and `--phased-phase2-data`. One or more wasn't correctly specified."
                )

            # if they both exist, must be Path objects
            if not (phased_phase1_data.exists() and phased_phase2_data.exists()):
                raise FileNotFoundError(
                    "Data for both phase1 and phase2 must be specified for phased training."
                )

            mt_bench_judge: pathlib.Path
            if phased_mt_bench_judge is None:
                ctx.fail(
                    "No MT-Bench model was provided with '--phased-mt-bench-judge'"
                )
            elif not phased_mt_bench_judge.resolve().exists():
                raise FileNotFoundError(
                    f"MT-Bench model directory could not be found: {phased_mt_bench_judge}\nMust be an absolute path to a model directory."
                )
            else:
                # makes MyPy happy because 'mt_bench_judge' isn't Path | None
                mt_bench_judge = phased_mt_bench_judge

            if not skip_user_confirm:
                click.confirm(
                    f"""\n\nSTARTING END-TO-END TRAINING.\nProceeding will OVERWRITE any metadata (checkpoints, intermediate results) that are stored at {phased_base_dir}.\n\nWould you like to proceed?""",
                    abort=True,
                )

            _prepare_phased_base_dir(phased_base_dir)

            _run_phased_training(
                ctx=ctx,
                train_args=train_args,
                torch_args=torch_args,
                base_dir=phased_base_dir,
                phase1_data=phased_phase1_data,
                phase1_num_epochs=phased_phase1_num_epochs,
                phase1_samples_per_save=phased_phase1_samples_per_save,
                phase1_checkpoints_dir=phased_base_dir / "phase1" / "checkpoints",
                phased_phase1_effective_batch_size=phased_phase1_effective_batch_size,
                phase2_data=phased_phase2_data,
                phase2_num_epochs=phased_phase2_num_epochs,
                phase2_samples_per_save=phased_phase2_samples_per_save,
                phase2_checkpoints_dir=phased_base_dir / "phase2" / "checkpoints",
                phased_phase2_effective_batch_size=phased_phase2_effective_batch_size,
                phase2_eval_cache=phased_base_dir / "phase2" / "eval_cache",
                mtbench_judge=mt_bench_judge,
                enable_serving_output=enable_serving_output,
            )

        else:
            if not os.path.isfile(data_path):
                ctx.fail(
                    f"Data path must be to a valid .jsonl file. Value given: {data_path}"
                )

            # TODO: should we have this wrapped in a try-except?
            run_training(train_args=train_args, torch_args=torch_args)


def _prepare_phased_base_dir(phased_base_dir: pathlib.Path) -> None:
    """Adds phase1 and phase2 directories in phased_base_dir.
    In each, adds `checkpoints` and `eval_cache` subdirectories.

    Args:
        phased_base_dir: directory wrapping phase1 and phase2 cached data.
    """

    logger.debug(f"Phased Training -- Preparing phased base dir: {phased_base_dir}")

    phase1_dir_path = phased_base_dir / "phase1"
    phase2_dir_path = phased_base_dir / "phase2"

    for p in [phase1_dir_path, phase2_dir_path]:
        utils.clear_directory(p)
        _setup_phase_dirs(p)


def _setup_phase_dirs(path: pathlib.Path) -> None:
    """Creates {path}/checkpoints and {path}/eval_cache directories."""

    # TODO: these sub-directory names are hard-coded here but they
    # could be parameterized in config.
    logger.debug(f"Phased Training -- Created phase directories for {path}")
    ckpt_path = path / "checkpoints"
    eval_cache_path = path / "eval_cache"

    os.makedirs(ckpt_path)
    os.makedirs(eval_cache_path)


def _training_phase(
    train_args: TrainingArgs,
    torch_args: TorchrunArgs,
    data_path: pathlib.Path,
    model_override: pathlib.Path | None = None,
    num_epochs: int | None = None,
    samples_per_save: int | None = None,
    checkpoint_dir: pathlib.Path | None = None,
    effective_batch_size: int | None = None,
) -> None:
    """A single step of phased training that supports key param overriding."""

    # Third Party
    from instructlab.training import run_training  # pylint: disable=no-name-in-module

    logger.debug(
        f"Phased Training -- training phase -- Overriding data_path: {train_args.data_path} with {data_path}"
    )

    # NOTE: have to cast pathlib.Path to str because Pydantic models require this. Here and below.
    train_args.data_path = str(data_path)

    if checkpoint_dir:
        train_args.ckpt_output_dir = str(checkpoint_dir)

    if model_override:
        logger.debug(
            f"Phased Training -- training phase -- Overriding model_path: {train_args.model_path} with {model_override}"
        )
        train_args.model_path = str(model_override)

    if num_epochs:
        logger.debug(
            f"Phased Training -- training phase -- Overriding num epochs: {train_args.num_epochs} with {num_epochs}"
        )
        train_args.num_epochs = num_epochs

    if samples_per_save is not None:
        logger.debug(
            f"Phased Training -- training phase -- Overriding samples per save: {train_args.save_samples} with {samples_per_save}"
        )
        train_args.save_samples = samples_per_save

    if effective_batch_size:
        logger.debug(
            f"Phased Training -- training phase -- Overriding effective batch size: {train_args.effective_batch_size} with {effective_batch_size}"
        )
        train_args.effective_batch_size = effective_batch_size

    click.secho(
        f"TrainingArgs for current phase: {pprint.pformat(train_args)}", fg="cyan"
    )

    run_training(train_args=train_args, torch_args=torch_args)


def _mmlu(model: pathlib.Path) -> float:
    # Third Party
    from instructlab.eval.mmlu import MMLU_TASKS, MMLUEvaluator
    import torch

    min_tasks = os.environ.get("INSTRUCTLAB_EVAL_MMLU_MIN_TASKS")
    if min_tasks is not None:
        tasks = ["mmlu_abstract_algebra", "mmlu_anatomy", "mmlu_astronomy"]
        evaluator = MMLUEvaluator(model, tasks=tasks)
    else:
        evaluator = MMLUEvaluator(model, tasks=MMLU_TASKS)

    # type the variable because MyPy doesn't seem to honor the types of the spread tuple
    ckpt_score: float
    ckpt_score, _ = evaluator.run()

    logging.debug("Phased Training -- MMLU eval phase -- Clearing PyTorch cache")
    torch.cuda.empty_cache()

    return ckpt_score


def _mtbench(
    ctx,
    model: pathlib.Path,
    eval_cache: pathlib.Path,
    mtbench_judge: pathlib.Path,
    enable_serving_output: bool,
) -> float:
    # TODO: optimization: run all generations in serial and then do all judgments at once to save time loading/unloading prometheus.

    # Third Party
    from instructlab.eval.mt_bench import MTBenchEvaluator
    import torch

    # First Party
    from instructlab.model.evaluate import launch_server

    # hard-override for local insecure vLLM serving
    ctx.params["tls_client_cert"] = None
    ctx.params["tls_client_key"] = None
    ctx.params["tls_client_passwd"] = None
    ctx.params["tls_insecure"] = True

    evaluator = MTBenchEvaluator(
        model_name=str(model),
        judge_model_name=str(mtbench_judge),
        output_dir=str(eval_cache),
        merge_system_user_message=True,  # TODO: expose this to the user
    )

    # TODO: Below, gpus=device_count() and max_workers=cpu_count() should be parameterized
    #   by the user.

    server = None
    model_serve_url = None
    try:
        logger.debug("Starting model server for mt-bench answer generation")
        server, model_serve_url = launch_server(
            ctx=ctx,
            model=str(model),
            model_name=str(model),
            gpus=torch.cuda.device_count(),
            max_workers=multiprocessing.cpu_count(),
            enable_serving_output=enable_serving_output,
            backend=backends.VLLM,
        )
        logger.debug("Generating mt-bench answers")
        evaluator.gen_answers(model_serve_url)
    finally:
        if server is not None:
            server.shutdown()

    try:
        logger.debug("Starting model server for mt-bench answer judgment")
        server, model_serve_url = launch_server(
            ctx=ctx,
            model=str(mtbench_judge),
            model_name=str(mtbench_judge),
            gpus=torch.cuda.device_count(),
            max_workers=multiprocessing.cpu_count(),
            backend=backends.VLLM,
            enable_serving_output=enable_serving_output,
        )
        logger.debug("Judging mt-bench answers")
        mt_bench_results: tuple = evaluator.judge_answers(model_serve_url)
        ckpt_score: float = mt_bench_results[0]
    finally:
        if server is not None:
            server.shutdown()

    return ckpt_score


def _evaluate_dir_of_checkpoints(
    checkpoints_dir: pathlib.Path, eval_func: typing.Callable[..., float]
) -> tuple[float, pathlib.Path]:
    """Run eval_func on all model checkpoints in a directory, returning the best performing."""

    # TODO: parallelize MMLU over available GPUs

    results: list[typing.Tuple[float, pathlib.Path]] = []
    for ckpt_path in checkpoints_dir.iterdir():
        if ckpt_path.is_dir():
            logger.debug(ckpt_path)
            ckpt_score = eval_func(model=ckpt_path)
            click.secho(
                f"CHECKPOINT EVALUATION: {ckpt_path} SCORED {ckpt_score}",
                fg="red",
                bg="cyan",
            )
            results.append((ckpt_score, ckpt_path))

    if not results:
        raise RuntimeError(
            "_evaluate_dir_of_checkpoints - No checkpoints were evaluated successfully at %s. No scores were recorded."
            % checkpoints_dir
        )

    ckpt_performance: list[tuple[float, pathlib.Path]] = sorted(results, reverse=True)

    # (score, path)
    return ckpt_performance[0]


def _run_phased_training(
    ctx,
    train_args: TrainingArgs,
    torch_args: TorchrunArgs,
    base_dir: pathlib.Path,
    phase1_data: pathlib.Path,
    phase1_num_epochs: int | None,
    phase1_samples_per_save: int | None,
    phase1_checkpoints_dir: pathlib.Path,
    phased_phase1_effective_batch_size: int | None,
    phase2_data: pathlib.Path,
    phase2_num_epochs: int | None,
    phase2_samples_per_save: int | None,
    phase2_checkpoints_dir: pathlib.Path,
    phased_phase2_effective_batch_size: int | None,
    phase2_eval_cache: pathlib.Path,
    mtbench_judge: pathlib.Path,
    enable_serving_output: bool,
) -> None:
    click.secho("Training Phase 1/2...", fg="cyan")

    _training_phase(
        train_args=train_args.model_copy(deep=True),
        torch_args=torch_args,
        data_path=phase1_data,
        checkpoint_dir=phase1_checkpoints_dir,
        num_epochs=phase1_num_epochs,
        samples_per_save=phase1_samples_per_save,
        effective_batch_size=phased_phase1_effective_batch_size,
        # model override not necessary because we expect model to come from ctx.params.model_path.
    )

    click.secho("MMLU evaluation for Phase 1...", fg="cyan")
    # NOTE: requires hf_format sub-directory. Training sets this up.
    phase1_checkpoints_dir = phase1_checkpoints_dir / "hf_format"
    _, p1_best_ckpt = _evaluate_dir_of_checkpoints(
        checkpoints_dir=phase1_checkpoints_dir, eval_func=_mmlu
    )

    click.secho("Training Phase 2/2...", fg="cyan")
    _training_phase(
        train_args=train_args.model_copy(deep=True),
        torch_args=torch_args,
        data_path=phase2_data,
        checkpoint_dir=phase2_checkpoints_dir,
        model_override=p1_best_ckpt,
        num_epochs=phase2_num_epochs,
        samples_per_save=phase2_samples_per_save,
        effective_batch_size=phased_phase2_effective_batch_size,
    )

    click.secho("MT-Bench evaluation for Phase 2...", fg="cyan")
    phase2_checkpoints_dir = phase2_checkpoints_dir / "hf_format"
    phase2_eval_cache = base_dir / "phase2" / "eval_cache"
    phase2_best_checkpoint_score, phase2_best_checkpoint = _evaluate_dir_of_checkpoints(
        checkpoints_dir=phase2_checkpoints_dir,
        eval_func=functools.partial(
            _mtbench,
            ctx=ctx,
            eval_cache=phase2_eval_cache,
            mtbench_judge=mtbench_judge,
            enable_serving_output=enable_serving_output,
        ),
    )

    click.secho(
        f"Training finished! Best final checkpoint: {phase2_best_checkpoint} with score: {phase2_best_checkpoint_score}",
        fg="green",
    )
