# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Optional
import logging
import os

# Third Party
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    TrainingArguments,
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
import click
import torch

# Local
from ..chat.chat import CONTEXTS

# TODO CPU: Look into using these extensions
# import intel_extension_for_pytorch as ipex

# Habana Labs framework for Intel Gaudi HPUs
# The imports register `hpu` device and `torch.hpu` package.
try:
    # pylint: disable=import-error
    # Third Party
    from habana_frameworks.torch import core as htcore
    from habana_frameworks.torch import hpu as hpu

    # Habana implementations of SFT Trainer
    # https://huggingface.co/docs/optimum/habana/index
    from optimum.habana import GaudiConfig, GaudiTrainingArguments
    from optimum.habana.transformers.generation.configuration_utils import (
        GaudiGenerationConfig,
    )
    from optimum.habana.trl import GaudiSFTTrainer

    # silence warning: "Call mark_step function will not have any effect"
    logging.getLogger("habana_frameworks.torch.utils.internal").setLevel(logging.ERROR)
except ImportError:
    htcore = None
    hpu = None
    hpu_backends = None


class StoppingCriteriaSub(StoppingCriteria):
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
    system: str = CONTEXTS["default"],
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


def report_hpu_device(args_device: torch.device) -> None:
    print(f"Device count: {hpu.device_count()}")
    for idx in range(hpu.device_count()):
        device = torch.device(args_device.type, idx)
        name: str = hpu.get_device_name(device)
        cap: str = hpu.get_device_capability(device)
        # property string is surrounded by '()'
        prop: str = hpu.get_device_properties(device)
        print(f"  {device} is '{name}', cap: {cap} {prop}")
    # https://docs.habana.ai/en/latest/PyTorch/Reference/Runtime_Flags.html
    print("PT and Habana Environment variables")
    for key, value in sorted(os.environ.items()):
        if key.startswith(("PT_", "HABANA", "LOG_LEVEL_", "ENABLE_CONSOLE")):
            print(f'  {key}="{value}"')


def pytorch_train(
    ctx: click.Context,
    train_file: str,
    test_file: str,
    model_name: str,
    num_epochs: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
    four_bit_quant: bool = False,
    dtype: str = "auto",
    train_style: str = "quick",
):
    """Lab Train for Linux and MacOS"""
    print("PYTORCH_TRAIN.PY: NUM EPOCHS IS: ", num_epochs)
    print("PYTORCH_TRAIN.PY: TRAIN FILE IS: ", train_file)
    print("PYTORCH_TRAIN.PY: TEST FILE IS: ", test_file)

    print(f"PYTORCH_TRAIN.PY: Using device '{device}'")
    torch.set_autocast_enabled(False)
    if device.type == "mps":
        print(f"MPS AVAILABLE: {torch.backends.mps.is_available()}")
        print(f"MPS BUILT: {torch.backends.mps.is_built()}")
    if device.type == "cuda":
        # estimated by watching nvtop / radeontop during training
        min_vram = 11 if four_bit_quant else 17

        # convert from gibibytes to bytes, torch.cuda.mem_get_info() returns bytes
        min_vram = min_vram * 1024**3

        report_cuda_device(device, min_vram)
    elif device.type == "hpu":
        if htcore is None:
            ctx.fail("habana_framework package is not installed.")
        if not hpu.is_available():
            ctx.fail("habana_framework is unable to detect HPUs.")
        hpu.init()
        report_hpu_device(device)

    print("PYTORCH_TRAIN.PY: LOADING DATASETS")
    # Get the file name
    train_dataset = load_dataset("json", data_files=train_file, split="train")

    test_dataset = load_dataset("json", data_files=test_file, split="train")
    train_dataset.to_pandas().head()

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
        print("PYTORCH_TRAIN.PY: USING 4-bit quantization with BitsAndBytes")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,  # if not set will throw a warning about slow speeds when training
        )
    else:
        print("PYTORCH_TRAIN.PY: NOT USING 4-bit quantization")
        bnb_config = None

    # Loading the model
    print("PYTORCH_TRAIN.PY: LOADING THE BASE MODEL")
    config = AutoConfig.from_pretrained(
        model_name, torchscript=True, trust_remote_code=True
    )

    torchDtype = None
    if dtype == "fp16":
        torchDtype = torch.float16
    if dtype == "bf16":
        torchDtype = torch.bfloat16
    if dtype == "fp32":
        torchDtype = torch.float32
    else:
        torchDtype = "auto"

    print(f"PYTORCH_TRAIN.PY: DOWNLOADING {model_name} TO RUN TRAINING.")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torchDtype,
        quantization_config=bnb_config,
        config=config,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print(f"PYTORCH_TRAIN.PY: DATATYPE IN USE: {model.dtype}")
    if model.device != device:
        print(
            f"PYTORCH_TRAIN.PY: MODEL DID NOT HAVE SAME DEVICE AS SPECIFIED. SWITCHING TO: {device}"
        )
        model = model.to(device)
    print(f"PYTORCH_TRAIN.PY: Model device {model.device}")
    if model.device.type == "cuda":
        print(torch.cuda.memory_summary())

    print("PYTORCH_TRAIN.PY: SANITY CHECKING THE BASE MODEL")
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
        text = create_prompt(user=user)

        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=256,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7,
            top_p=0.9,
            stopping_criteria=stopping_criteria,
            do_sample=True,
            **kwargs,
        )
        return tokenizer.batch_decode([o[:-1] for o in outputs])[0]

    # TODO: INFERENCING IS BROKEN ON MACOS, CANNOT INF ON CPU AND TRAIN ON GPU
    # MPS seems to be slow on this part. But, if the user is doing CPU or CUDA, this should still happen.
    assistant_old_lst = []
    if device.type != "mps":
        assistant_old_lst = [
            model_generate(d["user"]).split(response_template.strip())[-1].strip()
            for d in test_dataset
        ]
    attention_layers = [
        module for module in model.modules() if "attention" in str(type(module)).lower()
    ]

    # TODO: investigate if we can use CPU for the sanity check and GPU for train.
    # original_device = model.device
    # if model.device != device:
    #     print(f"here {device}")
    #     model = model.to(device)
    # print(f"PYTORCH_TRAIN.PY: Model device {model.device}")
    # if model.device.type == "cuda":
    #     print(torch.cuda.memory_summary())

    print("PYTORCH_TRAIN.PY: GETTING THE ATTENTION LAYERS")
    # Print information about the attention modules
    for i, layer in enumerate(attention_layers):
        for par in list(layer.named_parameters()):
            mod = par[0]
            if isinstance(mod, str):
                mod.split(".")[0]
        break

    print("PYTORCH_TRAIN.PY: CONFIGURING LoRA")

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
    output_dir = "./training_results"
    per_device_train_batch_size = 1

    # torch compile only for MPS
    # you need to rebuild the backend to get MPS enablement properly set up.
    if device.type == "mps":
        torch_compile = True
        torch_backend = "aot_eager"
    else:
        torch_compile = False
        torch_backend = None

    # change some options based off of train type
    if train_style == "quick":
        do_eval = False
        evaluation_strategy = "no"
        load_best_model_at_end = False
    elif train_style == "full":
        do_eval = True
        evaluation_strategy = "epoch"
        load_best_model_at_end = True
    else:
        raise ValueError(train_style)
    fp16 = dtype == "fp16"
    bf16 = not fp16 and device.type != "mps"
    max_seq_length = 300

    # TODO: This to @cdoern seems like it needs to be a different function, file, etc. Too different from the rest of train. Or at least, add flags for a bunch of this.
    if device.type == "hpu":
        # Intel Gaudi trainer
        # https://docs.habana.ai/en/latest/PyTorch/Getting_Started_with_PyTorch_and_Gaudi/Getting_Started_with_PyTorch.html
        # https://huggingface.co/docs/optimum/habana/quickstart
        # https://huggingface.co/docs/optimum/habana/package_reference/gaudi_config
        if per_device_train_batch_size == 1:
            per_device_train_batch_size = 8

        training_arguments = GaudiTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            bf16=True,
            save_strategy="epoch",
            report_to="none",
            use_habana=True,
            use_lazy_mode=True,
            # create checkpoint directories
            save_on_each_node=True,
            # gaudi_config_name=gaudi_config_name,
        )
        gaudi_config = GaudiConfig(
            use_fused_adam=True,
            use_fused_clip_norm=True,
            use_torch_autocast=True,
        )
        trainer = GaudiSFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            peft_config=peft_config,
            formatting_func=formatting_prompts_func,
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            args=training_arguments,
            gaudi_config=gaudi_config,
        )
        generate_kwargs = {
            # TODO: check generation config parameters?
            "generation_config": GaudiGenerationConfig(),
        }
    else:
        training_arguments = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            fp16=fp16,
            save_total_limit=2,
            load_best_model_at_end=load_best_model_at_end,
            optim="adamw_torch",
            evaluation_strategy=evaluation_strategy,
            do_eval=do_eval,
            bf16=bf16,
            use_cpu=model.device.type == "cpu",
            save_strategy="epoch",
            report_to="none",
            torch_compile_backend=torch_backend,
            torch_compile=torch_compile,
            # half_precision_backend = "cpu_amp",
            # use_ipex=True,
            # TODO CPU test this possible optimization
            #  gradient_accumulation_steps=4,
            #  gradient_checkpointing=True,
            # eval_accumulation_steps=1,
            # per_device_eval_batch_size=1,
            # bf16_full_eval=True,
            # options to reduce GPU memory usage and improve performance
            # https://huggingface.co/docs/transformers/perf_train_gpu_one
            # https://stackoverflow.com/a/75793317
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            peft_config=peft_config,
            formatting_func=formatting_prompts_func,
            data_collator=collator,
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            args=training_arguments,
        )
        generate_kwargs = {}

    print("PYTORCH_TRAIN.PY: TRAINING")
    trainer.train()

    model.config.use_cache = True
    print("PYTORCH_TRAIN.PY: RUNNING INFERENCE ON THE OUTPUT MODEL")

    for i, (d, assistant_old) in enumerate(zip(test_dataset, assistant_old_lst)):
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

    print("PYTORCH_TRAIN.PY: MERGING ADAPTERS")
    print(f"BEST CHECKPOINT TO BE USED: {trainer.state.best_model_checkpoint}")
    model = trainer.model.merge_and_unload()
    model.save_pretrained("./training_results/merged_model")
    print("PYTORCH_TRAIN.PY: FINISHED")
    return trainer.state.best_model_checkpoint
