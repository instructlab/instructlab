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
@click.option("--data-dir", help="Base directory where data is stored.", default=None)
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
    "--gguf-model-path",
    help="Local directory where gguf model is stored.",
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
    "--model-name",
    default="instructlab/merlinite-7b-lab",
    show_default=True,
    help="model name to use in training",
)
@click.pass_context
@utils.display_params
def train(
    ctx,
    data_dir,
    input_dir,
    skip_preprocessing,
    tokenizer_dir,
    gguf_model_path,
    model_dir,
    iters,
    local,
    skip_quantize,
    num_epochs,
    device: "torch.device",
    four_bit_quant: bool,
    model_name: str,
):
    """
    Takes synthetic data generated locally with `ilab generate` and the previous model and learns a new model using the MLX API.
    On success, writes newly learned model to {model_dir}/mlx_model, which is where `chatmlx` will look for a model.
    """
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

    # pylint: disable=import-outside-toplevel
    if not utils.is_macos_with_m_chip():
        # Local
        from ..llamacpp.llamacpp_convert_to_gguf import convert_llama_to_gguf
        from ..train.linux_train import linux_train

        training_results_dir = linux_train(
            ctx=ctx,
            train_file=train_file,
            test_file=test_file,
            model_name=model_name,
            num_epochs=num_epochs,
            device=device,
            four_bit_quant=four_bit_quant,
        )

        final_results_dir = training_results_dir / "final"
        if final_results_dir.exists():
            shutil.rmtree(final_results_dir)
        final_results_dir.mkdir()

        gguf_models_dir = Path("./models")
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
        # Local
        from ..mlx_explore.gguf_convert_to_mlx import load
        from ..mlx_explore.utils import fetch_tokenizer_from_hub
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

        # TODO copy some downloaded files from the PyTorch model folder
        # Seems to be not a problem if working with a remote download with convert.py
