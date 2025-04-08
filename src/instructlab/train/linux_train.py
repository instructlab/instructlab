# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
import logging
import os
import typing

# Third Party
# https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset
from datasets import load_dataset

# Parameter-Efficient Fine-Tuning: https://huggingface.co/docs/peft
from peft import LoraConfig
from tqdm import tqdm

# https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoConfig
# https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForCausalLM
# https://huggingface.co/docs/transformers/main_classes/quantization#transformers.BitsAndBytesConfig
# https://huggingface.co/docs/transformers/internal/generation_utils#transformers.StoppingCriteria
# https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)

# Transformer Reinforcement Learning
# https://huggingface.co/docs/trl/index
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer
import torch

# First Party
from instructlab.utils import ensure_legacy_dataset

# Local
from ..model.chat import CONTEXTS

logger = logging.getLogger(__name__)
# TODO CPU: Look into using these extensions
# import intel_extension_for_pytorch as ipex


class StoppingCriteriaSub(StoppingCriteria):
    # pylint: disable=unused-argument
    def __init__(self, stops=(), encounters=1, *, device: torch.device):
        super().__init__()
        self.device = device
        self.stops = [stop.to(device) for stop in stops]

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for seqs in input_ids:
            seq = seqs[-1].to(self.device)
            for stop in self.stops:
                if stop == seq:
                    return True
        return False


def create_prompt(
    user: str,
    system: str,
):
    return f"""\
    <|system|>
    {system}
    <|user|>
    {user}
    <|assistant|>
    """


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example["system"])):
        text = f"<|system|>\n{example['system'][i]}\n<|user|>\n{example['user'][i]}\n<|assistant|>\n{example['assistant'][i]}<|endoftext|>"
        output_texts.append(text)
    return output_texts


def report_cuda_device(args_device: torch.device, min_vram: int = 0) -> None:
    """Report CUDA/ROCm device properties"""
    print(f"  NVidia CUDA version: {torch.version.cuda or 'n/a'}")
    print(f"  AMD ROCm HIP version: {torch.version.hip or 'n/a'}")

    def _gib(size: int) -> str:
        return "{:.1f} GiB".format(size / 1024**3)

    for idx in range(torch.cuda.device_count()):
        device = torch.device("cuda", idx)
        name = torch.cuda.get_device_name(device)
        free, total = torch.cuda.mem_get_info(device)
        capmin, capmax = torch.cuda.get_device_capability(device)
        print(
            f"  {device} is '{name}' ({_gib(free)} of {_gib(total)} free, "
            f"capability: {capmin}.{capmax})"
        )

    if args_device.index is None:
        index = torch.cuda.current_device()
    else:
        index = args_device.index

    free = torch.cuda.mem_get_info(index)[0]
    if free < min_vram:
        print(
            f"  WARNING: You have less than {min_vram} GiB of free GPU "
            "memory on '{index}'. Training may fail, use slow shared "
            "host memory, or move some layers to CPU."
        )
        print(
            "  Training does not use the local InstructLab serve. Consider "
            "stopping the server to free up about 5 GiB of GPU memory."
        )


def linux_train(
    train_file: Path,
    test_file: Path,
    model_name: str,
    num_epochs: int | None,
    train_device: str = "cpu",
    four_bit_quant: bool = False,
    output_dir: Path = Path("training_results"),
) -> Path:
    """Lab Train for Linux!"""

    try:
        device = torch.device(train_device)
    except RuntimeError as e:
        raise RuntimeError(f"Error getting Torch Device {str(e)}") from e
    # Detect CUDA/ROCm device
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"{device.type}: Torch has no CUDA/ROCm support or could not detect a compatible device."
            )
        # map unqualified 'cuda' to current device
        if device.index is None:
            device = torch.device(device.type, torch.cuda.current_device())

    if device.type == "hpu":
        raise RuntimeError(
            f"{device.type}: Simple training does not support Intel Gaudi."
        )

    if four_bit_quant and device.type != "cuda":
        raise RuntimeError("'--4-bit-quant' option requires '--device=cuda'")

    print("LINUX_TRAIN.PY: NUM EPOCHS IS: ", num_epochs)
    print("LINUX_TRAIN.PY: TRAIN FILE IS: ", train_file)
    print("LINUX_TRAIN.PY: TEST FILE IS: ", test_file)

    print(f"LINUX_TRAIN.PY: Using device '{device}'")
    output_dir = Path(output_dir)
    if device.type == "cuda":
        # estimated by watching nvtop / radeontop during training
        min_vram = 11 if four_bit_quant else 17

        # convert from gibibytes to bytes, torch.cuda.mem_get_info() returns bytes
        min_vram = min_vram * 1024**3

        report_cuda_device(device, min_vram)

    print("LINUX_TRAIN.PY: LOADING DATASETS")
    # Get the file name
    train_dataset = load_dataset(
        "json", data_files=os.fspath(train_file), split="train"
    )

    test_dataset = load_dataset("json", data_files=os.fspath(test_file), split="train")
    try:
        train_dataset = ensure_legacy_dataset(train_dataset)
        test_dataset = ensure_legacy_dataset(test_dataset)
    except ValueError as e:
        raise ValueError(f"Failed to parse dataset: {e}") from e

    train_dataset.to_pandas().head()

    # https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoTokenizer.from_pretrained
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    response_template = "\n<|assistant|>\n"

    response_template_ids = tokenizer.encode(
        response_template, add_special_tokens=False
    )[2:]
    collator = DataCollatorForCompletionOnlyLM(
        response_template_ids, tokenizer=tokenizer
    )

    if four_bit_quant:
        print("LINUX_TRAIN.PY: USING 4-bit quantization with BitsAndBytes")
        use_fp16 = True
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,  # if not set will throw a warning about slow speeds when training
        )
    else:
        print("LINUX_TRAIN.PY: NOT USING 4-bit quantization")
        use_fp16 = False
        bnb_config = None

    # Loading the model
    print("LINUX_TRAIN.PY: LOADING THE BASE MODEL")
    config = AutoConfig.from_pretrained(
        model_name, torchscript=True, trust_remote_code=True
    )

    # https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoModelForCausalLM.from_pretrained
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        quantization_config=bnb_config,
        config=config,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if model.device != device:
        model = model.to(device)
    print(f"LINUX_TRAIN.PY: Model device {model.device}")
    if model.device.type == "cuda":
        print(torch.cuda.memory_summary())

    print("LINUX_TRAIN.PY: SANITY CHECKING THE BASE MODEL")
    stop_words = ["<|endoftext|>", "<|assistant|>"]
    stop_words_ids = [
        tokenizer(stop_word, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ].squeeze()
        for stop_word in stop_words
    ]
    stopping_criteria = StoppingCriteriaList(
        [StoppingCriteriaSub(stops=stop_words_ids, device=model.device)]
    )

    def model_generate(user, **kwargs):
        system_prompt = CONTEXTS["default"]("default")
        text = create_prompt(user=user, system=system_prompt)

        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=256,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7,
            top_p=0.9,
            stopping_criteria=stopping_criteria,
            do_sample=True,
            output_logits=True,
            **kwargs,
        )
        return tokenizer.batch_decode([o[:-1] for o in outputs])[0]

    assistant_old_lst = [
        model_generate(d["user"]).split(response_template.strip())[-1].strip()
        for d in tqdm(test_dataset)
    ]
    attention_layers = [
        module for module in model.modules() if "attention" in str(type(module)).lower()
    ]

    print("LINUX_TRAIN.PY: GETTING THE ATTENTION LAYERS")
    # Print information about the attention modules
    for layer in attention_layers:
        for par in list(layer.named_parameters()):
            mod = par[0]
            if isinstance(mod, str):
                mod = mod.split(".")[0]
        break

    print("LINUX_TRAIN.PY: CONFIGURING LoRA")

    lora_alpha = 32
    lora_dropout = 0.1
    lora_r = 4

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    tokenizer.padding_side = "right"
    per_device_train_batch_size = 1
    max_seq_length = 300
    generate_kwargs: dict[str, typing.Any]

    training_arguments = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        fp16=use_fp16,
        bf16=not use_fp16,
        # use_ipex=True, # TODO CPU test this possible optimization
        use_cpu=model.device.type == "cpu",
        save_strategy="epoch",
        report_to="none",
        max_seq_length=max_seq_length,
        # options to reduce GPU memory usage and improve performance
        # https://huggingface.co/docs/transformers/perf_train_gpu_one
        # https://stackoverflow.com/a/75793317
        # torch_compile=True,
        # fp16=False,  # fp16 increases memory consumption 1.5x
        # gradient_accumulation_steps=8,
        # gradient_checkpointing=True,
        # eval_accumulation_steps=1,
        # per_device_eval_batch_size=1,
    )

    # https://huggingface.co/docs/trl/main/en/sft_trainer#trl.SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        processing_class=tokenizer,
        args=training_arguments,
    )
    generate_kwargs = {}

    print("LINUX_TRAIN.PY: TRAINING")
    trainer.train()

    model.config.use_cache = True

    print("LINUX_TRAIN.PY: RUNNING INFERENCE ON THE OUTPUT MODEL")

    for i, (d, assistant_old) in enumerate(
        zip(test_dataset, assistant_old_lst, strict=False)
    ):
        output = model_generate(d["user"], **generate_kwargs)
        assistant_new = output.split(response_template.strip())[-1].strip()
        assistant_expected = d["assistant"]

        print(f"\n===\ntest {i}\n===\n")
        print("\n===\nuser\n===\n")
        print(d["user"])
        print("\n===\nassistant_old\n===\n")
        print(assistant_old)
        print("\n===\nassistant_new\n===\n")
        print(assistant_new)
        print("\n===\nassistant_expected\n===\n")
        print(assistant_expected)

    print("LINUX_TRAIN.PY: MERGING ADAPTERS")
    model = trainer.model.merge_and_unload()
    model.save_pretrained(output_dir / "merged_model")

    print("LINUX_TRAIN.PY: FINISHED")
    return output_dir
