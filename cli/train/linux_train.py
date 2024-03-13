# SPDX-FileCopyrightText: The InstructLab Authors
# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
import argparse

# Third Party
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TrainingArguments,
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
import torch

# First Party
from cli.chat.chat import CONTEXTS

# TODO CPU: Look into using these extensions
# import intel_extension_for_pytorch as ipex


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        # TODO GPU: the commented out version might be needed to work on Nvidia GPUs
        # self.stops = [stop.to("cuda") for stop in stops]
        self.stops = [stop for stop in stops]

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for seq in input_ids:
            for stop in self.stops:
                if stop == seq[-1]:
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


def main(args_in: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Lab Train for Linux!")
    parser.add_argument(
        "--train-file",
        type=str,
        help="absolute path to the training file",
        default=None,
    )
    parser.add_argument(
        "--test-file", type=str, help="absolute path to the testing file", default=None
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        help="number of epochs to run during training",
        default=None,
    )
    args = parser.parse_args(args_in)

    print("LINUX_TRAIN.PY: NUM EPOCHS IS: ", args.num_epochs)
    print("LINUX_TRAIN.PY: TRAIN FILE IS: ", args.train_file)
    print("LINUX_TRAIN.PY: TEST FILE IS: ", args.test_file)

    print("LINUX_TRAIN.PY: LOADING DATASETS")
    # Get the file name
    train_dataset = load_dataset("json", data_files=args.train_file, split="train")

    test_dataset = load_dataset("json", data_files=args.test_file, split="train")
    train_dataset.to_pandas().head()

    model_name = "ibm/merlinite-7b"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    response_template = "\n<|assistant|>\n"

    response_template_ids = tokenizer.encode(
        response_template, add_special_tokens=False
    )[2:]
    collator = DataCollatorForCompletionOnlyLM(
        response_template_ids, tokenizer=tokenizer
    )

    # TODO GPU: bnb_config is needed quantize on Nvidia GPUs
    # bnb_config = BitsAndBytesConfig(
    #    load_in_4bit=True,
    #    bnb_4bit_quant_type="nf4",
    #    bnb_4bit_use_double_quant=True,
    #    bnb_4bit_compute_dtype=torch.float16 # if not set will throw a warning about slow speeds when training
    # )

    # Loading the model
    print("LINUX_TRAIN.PY: LOADING THE BASE MODEL")
    config = AutoConfig.from_pretrained(
        model_name, torchscript=True, trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        # quantization_config=bnb_config,
        config=config,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    print("LINUX_TRAIN.PY: SANITY CHECKING THE BASE MODEL")
    stop_words = ["<|endoftext|>", "<|assistant|>"]
    stop_words_ids = [
        tokenizer(stop_word, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ].squeeze()
        for stop_word in stop_words
    ]
    stopping_criteria = StoppingCriteriaList(
        [StoppingCriteriaSub(stops=stop_words_ids)]
    )

    def model_generate(user):
        text = create_prompt(user=user)

        # TODO GPU: the commented out line might be needed to work on Nvidia GPUs
        # input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
        input_ids = tokenizer(text, return_tensors="pt").input_ids
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=256,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7,
            top_p=0.9,
            stopping_criteria=stopping_criteria,
            do_sample=True,
        )
        return tokenizer.batch_decode([o[:-1] for o in outputs])[0]

    model_generate(
        "In excruciating detail, explain to me the nuances of who runs Barter Town."
    )
    assistant_old_lst = [
        model_generate(d["user"]).split(response_template.strip())[-1].strip()
        for d in test_dataset
    ]
    attention_layers = [
        module for module in model.modules() if "attention" in str(type(module)).lower()
    ]

    print("LINUX_TRAIN.PY: GETTING THE ATTENTION LAYERS")
    # Print information about the attention modules
    for i, layer in enumerate(attention_layers):
        for par in list(layer.named_parameters()):
            mod = par[0]
            if isinstance(mod, str):
                mod.split(".")[0]
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

    output_dir = "./training_results"
    per_device_train_batch_size = 1

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        bf16=True,
        # use_ipex=True, # TODO CPU test this possible optimization
        use_cpu=True,
        save_strategy="epoch",
        report_to="none",
    )

    max_seq_length = 300
    tokenizer.padding_side = "right"
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

    print("LINUX_TRAIN.PY: TRAINING")
    trainer.train()

    model.config.use_cache = True

    print("LINUX_TRAIN.PY: RUNNING INFERENCE ON THE OUTPUT MODEL")

    for (i, (d, assistant_old)) in enumerate(zip(test_dataset, assistant_old_lst)):
        assistant_new = (
            model_generate(d["user"]).split(response_template.strip())[-1].strip()
        )
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
    model.save_pretrained("./training_results/merged_model")

    print("LINUX_TRAIN.PY: FINISHED")


if __name__ == "__main__":
    main()
