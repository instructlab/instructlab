# SPDX-License-Identifier: Apache-2.0

# Standard
from glob import glob
from pathlib import Path
import os
import shutil

# Third Party
import click
import torch

# First Party
from instructlab import utils
from instructlab_train import torchrun_train

class TorchDeviceParam(click.ParamType):
    """Parse and convert device string

    Returns a torch.device object:
    - type is one of 'cpu', 'cuda', 'hpu'
    - index is None or device index (e.g. 0 for first GPU)
    """

    name = "deviceinfo"
    supported_devices = {"cuda", "cpu", "hpu"}

    def convert(self, value, param, ctx) -> "torch.device":
        # pylint: disable=C0415
        # Function local import, import torch can take more than a second
        # Third Party
        import torch

        if not isinstance(value, torch.device):
            try:
                device = torch.device(value)
            except RuntimeError as e:
                self.fail(str(e), param, ctx)

        if device.type not in self.supported_devices:
            supported = ", ".join(repr(s) for s in sorted(self.supported_devices))
            self.fail(
                f"Unsupported device type '{device.type}'. Only devices "
                f"types {supported}, and indexed device strings like 'cuda:0' "
                "are supported for now.",
                param,
                ctx,
            )

        # Detect CUDA/ROCm device
        if device.type == "cuda":
            if not torch.cuda.is_available():
                self.fail(
                    f"{value}: Torch has no CUDA/ROCm support or could not detect "
                    "a compatible device.",
                    param,
                    ctx,
                )
            # map unqualified 'cuda' to current device
            if device.index is None:
                device = torch.device(device.type, torch.cuda.current_device())

        if device.type == "hpu":
            click.secho(
                "WARNING: HPU support is experimental, unstable, and not "
                "optimized, yet.",
                fg="red",
                bold=True,
            )

        return device


TORCH_DEVICE = TorchDeviceParam()


@click.command()
@click.option(
    "--data-dir", 
    help="Base directory where data is stored.", 
    default=None
)
@click.option(
    "--input-dir",
    type=click.Path(),
    show_default=True,  # TODO: set to None and change help message
    help="Path to generated files to use as input.",
)
@click.option(
    "--skip-preprocessing",
    is_flag=True,
)
@click.option(
    "--tokenizer-dir",
    help="Base directory where tokenizer is stored.",
    default=None,
    show_default=True,
)
@click.option(
    "--model-dir",
    help="Base directory where model is stored.",
    default="instructlab/merlinite-7b-lab",
    show_default=True,
)
@click.option("--iters", help="Number of iterations to train LoRA.", default=100)
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
    default=1,  # TODO: change this to a more reasonable default
    show_default=True,
    help="The number of times the training data is passed through the training algorithm. Please note that this value is used on Linux platforms only.",
)
@click.option(
    "--device",
    type=TORCH_DEVICE,
    show_default=True,
    default="cpu",
    help=(
        "PyTorch device for Linux training (default: 'cpu'). Use 'cuda' "
        "for NVidia CUDA / AMD ROCm GPU, 'cuda:0' for first GPU."
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
    "--gpus",
    type=str,
    default="-1",
    help="GPUs to use for training"

)
@click.option(
    "--max-seq-len",
    type=int
)
@click.option(
    "--max-batch-len",
    type=int
)
@click.option(
    "--effective-batch-size",
    type=int
)
@click.option(
    "--save-samples",
    type=int
)
@click.option(
    "--learning-rate",
    "lr",
    type=float
)
@click.option(
    "--warmup-steps",
    type=int
)
@click.option(
    "--deepspeed",
    type=bool
)
@click.option(
    "--deepspeed-config",
    type=click.Path
)
@click.option(
    "--offload-strategy",
    type=click.Choice(["cpu", "nvme", None]),
    default=None
)
# these two seem like they could be inferred by the above?
@click.option(
    "--cpu-offload-optim",
    type=bool,
)
@click.option(
    "--cpu-offload-params",
    type=bool
)
@click.option(
    "--ds-quantize-type",
    type=click.Choice(["nf4", "fp8", None]),
    default=None
)
@click.option(
    "--lora",
    type=bool,
    default=False
)
# below flags are invalid if lora == false
@click.option(
    "--lora-rank",
    type=int
)
@click.option(
    "--lora-alpha",
    type=float
)
@click.option(
    "--lora-dropout",
    type=float
)
@click.option(
    "--target-modules",
    type=[],
)
@click.pass_context
def train(
    ctx,
    data_dir,
    input_dir,
    skip_preprocessing,
    tokenizer_dir,
    model_dir,
    iters,
    local,
    skip_quantize,
    num_epochs,
    device: "torch.device",
    four_bit_quant: bool,
    gpus,
    max_seq_len,
    max_batch_len,
    effective_batch_size,
    save_samples,
    learning_rate,
    warmup_steps,
    deepspeed,
    deepspeed_config,
    offload_strategy,
    cpu_offload_optim,
    cpu_offload_params,
    ds_quantize_dtype,
    lora,
    lora_rank,
    lora_dropout,
    target_modules,
):
    """
    Takes synthetic data generated locally with `ilab generate` and the previous model and learns a new model using the MLX API.
    On success, writes newly learned model to {model_dir}/mlx_model, which is where `chatmlx` will look for a model.
    """

    # how do we differentiate between usecases?

    if not input_dir:
        # By default, generate output-dir is used as train input-dir
        input_dir = ctx.obj.config.generate.output_dir

    if four_bit_quant and device.type != "cuda":
        ctx.fail("--4-bit-quant option requires --device=cuda")

    effective_data_dir = Path(data_dir or "./taxonomy_data")
    train_file = effective_data_dir / "train_gen.jsonl"
    test_file = effective_data_dir / "test_gen.jsonl"

    # NOTE: If given a data_dir, input-dir is ignored in favor of existing!
    if data_dir is None:
        data_dir = effective_data_dir
        if not os.path.exists(input_dir):
            click.secho(
                f"Could not read directory: {input_dir}",
                fg="red",
            )
            raise click.exceptions.Exit(1)

        try:
            os.makedirs(data_dir, exist_ok=True)
        except OSError as exc:
            click.secho(
                f"Could not create data dir: {exc}",
                fg="red",
            )
            raise click.exceptions.Exit(1)

        # generated input files reverse sorted by name (contains timestamp)
        def get_files(pattern):
            return sorted(Path(input_dir).glob(pattern), reverse=True)

        train_files = get_files("train_*")
        test_files = get_files("test_*")

        if not train_files or not test_files:
            click.secho(
                f"{input_dir} does not contain training or test files, did you run `ilab generate`?",
                fg="red",
            )
            raise click.exceptions.Exit(1)
        if len(train_files) > 1 or len(test_files) > 1:
            click.secho(
                "Found multiple files from `ilab generate`. Using the most recent generation.",
                fg="yellow",
            )
        # First file is latest (by above reverse sort and timestamped names)
        shutil.copy(train_files[0], train_file)
        shutil.copy(test_files[0], test_file)

    # if macos, preserve that path
    if utils.is_macos_with_m_chip():
        # Local
        from ..train.lora_mlx.convert import convert_between_mlx_and_pytorch
        from ..train.lora_mlx.lora import load_and_train
        from ..train.lora_mlx.make_data import make_data

        if not skip_preprocessing:
            try:
                make_data(data_dir=data_dir)
            except FileNotFoundError as exc:
                click.secho(
                    f"Could not read from data directory: {exc}",
                    fg="red",
                )
                raise click.exceptions.Exit(1)

        # NOTE we can skip this if we have a way ship MLX
        # PyTorch safetensors to MLX safetensors
        model_dir_local = model_dir.replace("/", "-")
        model_dir_mlx = f"{model_dir_local}-mlx"
        model_dir_mlx_quantized = f"{model_dir_local}-mlx-q"

        if skip_quantize:
            dest_model_dir = model_dir_mlx
            quantize_arg = False
        else:
            dest_model_dir = model_dir_mlx_quantized
            quantize_arg = True


        # Downloading PyTorch SafeTensor and Converting to MLX SafeTensor
        convert_between_mlx_and_pytorch(
            hf_path=model_dir,
            mlx_path=dest_model_dir,
            quantize=quantize_arg,
            local=local,
        )

        adapter_file_path = f"{dest_model_dir}/adapters.npz"
        # train the model with LoRA

        load_and_train(
            model=dest_model_dir,
            train=True,
            data=data_dir,
            adapter_file=adapter_file_path,
            iters=iters,
            save_every=10,
            steps_per_eval=10,
        )
    else:
        # take flags, funnel them into a _train object, pass it to library.
    #   execute library code
        torchrun_train(
            # somehow pass all above flags
            # torch args too
        )
