# Standard
from pathlib import Path
import logging
import os
import pathlib
import shutil

# First Party
from instructlab import utils
from instructlab.configuration import DEFAULTS

# pylint: disable=ungrouped-imports

logger = logging.getLogger(__name__)


def simple_train(
    model_path,
    skip_preprocessing,
    skip_quantize,
    gguf_model_path,
    tokenizer_dir,
    data_path,
    input_dir,
    ckpt_output_dir,
    iters,
    local,
    num_epochs,
    device,
    four_bit_quant,
):
    effective_data_dir: pathlib.Path = Path(
        data_path if data_path else DEFAULTS.DATASETS_DIR
    )

    train_file = effective_data_dir / "train_gen.jsonl"
    test_file = effective_data_dir / "test_gen.jsonl"

    # NOTE: If given a data_dir, input-dir is ignored in favor of existing!
    if not data_path or data_path.strip() == DEFAULTS.DATASETS_DIR:
        data_path = str(effective_data_dir)
        if not os.path.exists(input_dir):
            raise OSError(
                f"Could not read directory: {input_dir}",
            )

        try:
            os.makedirs(data_path, exist_ok=True)
        except OSError as exc:
            raise OSError(f"Could not create data dir: {exc}") from exc

        # generated input files reverse sorted by modification time
        def get_files(directory: str, pattern: str) -> list[str]:
            return sorted(
                [str(p) for p in Path(directory).glob(pattern)],
                key=os.path.getmtime,
                reverse=True,
            )

        # Find applicable test and train files in the datasets directory.
        # Search in both per-run directories as well as top-level for backwards compatibility.
        # ignore the test_file and train_file to prevent it from being copied back onto itself
        # see: https://github.com/instructlab/instructlab/pull/1685
        test_files = [
            f
            for f in get_files(input_dir, "*/test_*") + get_files(input_dir, "test_*")
            if os.path.basename(f) != os.path.basename(test_file)
        ]
        train_files = [
            f
            for f in get_files(input_dir, "*/train_*") + get_files(input_dir, "train_*")
            if os.path.basename(f) != os.path.basename(train_file)
        ]

        if not train_files or not test_files:
            raise FileNotFoundError(
                f"{input_dir} does not contain training or test files, did you run `ilab data generate`?",
            )
        if len(train_files) > 1 or len(test_files) > 1:
            logger.warning(
                "Found multiple files from `ilab data generate`. Using the most recent generation.",
            )
        # The first file is latest
        logger.debug("train_file=%s", train_files[0])
        logger.debug("test_file=%s", test_files[0])
        shutil.copy(train_files[0], train_file)
        shutil.copy(test_files[0], test_file)

    if utils.is_macos_with_m_chip():
        # Local
        from ..llamacpp.convert_to_gguf import convert_model_to_gguf
        from ..mlx_explore.gguf_convert_to_mlx import load
        from ..mlx_explore.utils import fetch_tokenizer_from_hub
        from ..train.lora_mlx.convert import convert_between_mlx_and_pytorch
        from ..train.lora_mlx.lora import load_and_train
        from ..train.lora_mlx.make_data import make_data

        if not skip_preprocessing:
            try:
                make_data(data_dir=data_path)
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"Could not read from data directory: {exc}"
                ) from exc

        # NOTE we can skip this if we have a way to ship MLX
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
            load(
                gguf=gguf_model_path,
                repo=tokenizer_dir,
                mlx_path=dest_model_dir,
            )

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

        source_model_dir = dest_model_dir
        model_dir_fused = f"{source_model_dir}-fused"
        convert_after_training(
            source_model_dir=source_model_dir,
            model_dir_fused=model_dir_fused,
            adapter_file_path=adapter_file_path,
            model_dir_local=model_dir_local,
        )

    else:
        # Local
        from ..llamacpp.convert_to_gguf import convert_model_to_gguf
        from ..train.linux_train import linux_train

        training_results_dir = linux_train(
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

        gguf_models_file = final_results_dir / "ggml-model-f16.gguf"
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
            logger.info(f"Copied {file_} to {final_results_dir}")

        for file in training_results_dir.glob("merged_model/*.safetensors"):
            shutil.move(file, final_results_dir)
            logger.info(f"Moved {file} to {final_results_dir}")

        if four_bit_quant:
            logger.info(
                "SKIPPING CONVERSION to gguf. This is unsupported with --4-bit-quant. "
                + "See https://github.com/instructlab/instructlab/issues/579."
            )
            return

        gguf_file_path = convert_model_to_gguf(model=final_results_dir)

        shutil.move(gguf_file_path, gguf_models_file)
        logger.info(f"Save trained model to {gguf_models_file}")

        # Remove safetensors files to save space, were done with them here
        # and the huggingface lib has them cached
        for file in final_results_dir.glob("*.safetensors"):
            file.unlink()

        shutil.move(gguf_file_path, gguf_models_file)
        logger.info(f"Save trained model to {gguf_models_file}")


def convert_after_training(
    source_model_dir, model_dir_fused, adapter_file_path, model_dir_local
):
    """
    convert_after_training is a utility used by simple training to convert a model to GGUF after training

    Args:
        source_model_dir: str
        model_dir_fused: str
        adapter_file_path: str
        model_dir_local: str
    """
    # Local
    from ..llamacpp.convert_to_gguf import convert_model_to_gguf
    from ..train.lora_mlx.convert import convert_between_mlx_and_pytorch
    from ..train.lora_mlx.fuse import fine_tune

    fine_tune(
        model=source_model_dir,
        save_path=model_dir_fused,
        adapter_file=adapter_file_path,
        de_quantize=True,
    )

    model_dir_fused_pt = f"{source_model_dir}-trained"
    # this converts MLX to PyTorch
    convert_between_mlx_and_pytorch(
        hf_path=model_dir_fused, mlx_path=model_dir_fused_pt, local=True, to_pt=True
    )

    gguf_file_path = convert_model_to_gguf(model=Path(model_dir_fused_pt))

    gguf_models_file = Path(f"{model_dir_local}.gguf")
    # Remove previously trained model, its taking up space we may need in the next step
    gguf_models_file.unlink(missing_ok=True)

    shutil.move(gguf_file_path, gguf_models_file)
    logger.info(f"Save trained model to {gguf_models_file}")

    shutil.rmtree(model_dir_fused_pt, ignore_errors=True)
    logger.info(f"Removed {model_dir_fused_pt}")

    shutil.rmtree(model_dir_fused, ignore_errors=True)
    logger.info(f"Removed {model_dir_fused}")

    shutil.rmtree(source_model_dir, ignore_errors=True)
    logger.info(f"Removed {source_model_dir}")
