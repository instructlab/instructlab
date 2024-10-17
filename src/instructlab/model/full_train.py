# Standard
from copy import copy
from functools import partial
from pathlib import Path
import logging
import math
import os

# Third Party
from instructlab_quantize import run_quantize
from tqdm import tqdm
from transformers import Adafactor, AutoModelForCausalLM
import numpy as np
import psutil

# First Party
from instructlab.llamacpp import llamacpp_convert_to_gguf

logger = logging.getLogger(__name__)


def train(train_args, device):
    """
    train runs a CPU and MacOS optimized version of full fine tuning.
    Adafactor is the optimizer of choice and the multiprocessing method is set to spawn.
    Dataloading functions imported from the training library.
    """

    # pylint: disable=no-name-in-module
    # Third Party
    from instructlab.training import config
    from instructlab.training import data_process as dp
    from instructlab.training import (
        multipack_sampler,
        token_dataset,
        tokenizer_utils,
        utils,
    )
    from torch.utils.data import DataLoader
    import torch

    dp.main(
        config.DataProcessArgs(
            data_output_path=train_args.data_output_dir,
            model_path=train_args.model_path,
            data_path=train_args.data_path,
            max_seq_len=train_args.max_seq_len,
            chat_tmpl_path=train_args.chat_tmpl_path,
        )
    )

    # load chat template based on path in the args
    CHAT_TEMPLATE, SPECIAL_TOKENS = utils.retrieve_chat_template(
        train_args.chat_tmpl_path
    )
    # get the tokenizer for the model
    tokenizer = tokenizer_utils.setup_tokenizer(
        train_args.model_path, SPECIAL_TOKENS, CHAT_TEMPLATE
    )

    # setup the dataset and place it in data.jsonl, this needs to be used for training NOT the jsonl produced by sdg
    dataset = token_dataset.setup_dataset(
        os.path.join(train_args.data_output_dir, "data.jsonl"),
    )

    # based on the length of the dataset, figure out the max batch len
    packing_max_batch_len, accum = (
        multipack_sampler.find_packing_max_batch_len_and_grad_accum(
            num_gpus=1,
            avg_sample_len=dataset.get_lengths().mean(),
            effective_batch_size=train_args.effective_batch_size,
            max_batch_len_per_gpu=train_args.max_batch_len,
            is_padding=True,
            dataset=dataset,
            seed=47,
        )
    )

    collate_fn = partial(pad_collate_fn, pad_token_id=tokenizer.pad_token_id)

    # use a multipack sampler since this is a non-dist. scenario, training library defaults to this as well
    sampler = multipack_sampler.MultipackDistributedBatchSampler(
        batch_max_length=packing_max_batch_len,
        lengths=dataset.get_lengths(),
        num_replicas=1,
        rank=0,
        seed=47,
        padding=True,
    )
    sampler = {"batch_sampler": sampler}

    # 4 workers for the dataloader as compared to the original 8 to optimize performance
    # this dataloader needs a spawn multiproc method to optimize memory and to work on Apple Silicon.
    dataloader = DataLoader(
        dataset,
        **sampler,
        num_workers=4,
        collate_fn=collate_fn,
    )
    dataloader.multiprocessing_context = "spawn"

    logger.info(
        f"avg_sample_len: {dataset.get_lengths().mean()}\n effective_batch_size: {train_args.effective_batch_size}\n max_batch_len: {train_args.max_batch_len}\n packing_max_batch_len: {packing_max_batch_len} \n grad_accum: {accum}\n  num_batches: {len(dataloader)}\n avg_samples_per_batch: {len(dataset) / len(dataloader)}"
    )
    # set device based off argument given
    dev = torch.device(device)
    # auto model based on model path
    model = AutoModelForCausalLM.from_pretrained(train_args.model_path)
    if len(tokenizer) > model.config.vocab_size:
        logger.warning(
            f"WARNING: tokenizer has {len(tokenizer)} tokens but model has {model.config.vocab_size} vocab size"
        )
        model.resize_token_embeddings(
            int(8 * math.ceil(len(tokenizer) / 8.0))
        )  # make the vocab size multiple of 8 for sharding the embedding layer.

    # Fix any discrepancy between model and tokenizer
    # bos and eos tokens mark the beginning and end of a sequence/sentence
    if (
        model.config.pad_token_id is not None
        and tokenizer.pad_token_id is not None
        and model.config.pad_token_id != tokenizer.pad_token_id
    ):
        logger.warning(
            f"There is a mismatch between pad token id of model ({model.config.pad_token_id}) and tokenizer ({tokenizer.pad_token_id}). Fixing model pad token id to be same as tokenizer's pad token id."
        )
        model.config.pad_token_id = tokenizer.pad_token_id
    if (
        model.config.bos_token_id is not None
        and tokenizer.bos_token_id is not None
        and model.config.bos_token_id != tokenizer.bos_token_id
    ):
        logger.warning(
            f"There is a mismatch between bos token id of model ({model.config.bos_token_id}) and tokenizer ({tokenizer.bos_token_id}). These tokens denote the start of a sequence of data. Fixing model bos token id to be same as tokenizer's bos token id."
        )
        model.config.bos_token_id = tokenizer.bos_token_id
    if (
        model.config.eos_token_id is not None
        and tokenizer.eos_token_id
        and model.config.eos_token_id != tokenizer.eos_token_id
    ):
        logger.warning(
            f"There is a mismatch between eos token id of model ({model.config.eos_token_id}) and tokenizer ({tokenizer.eos_token_id}). These tokens denote the end of a sequence of data. Fixing model eos token id to be same as tokenizer's eos token id."
        )
        model.config.eos_token_id = tokenizer.eos_token_id

    model = utils.convert_loss_to_reduce_sum(model)
    model = utils.add_noisy_embeddings(model, noise_alpha=None)

    # Get virtual memory statistics
    memory_info = psutil.virtual_memory()

    # Total RAM
    total_ram = memory_info.total / (1024**3)  # Convert to GB
    logger.info(f"Total RAM: {total_ram:.2f} GB")
    # if RAM is <= 16, we need to use fp16 not fp32. This will yield a worse model but will allow the full pipeline to run
    if total_ram <= 16:
        # if <= 16GB ram, use gradinent accum and hald precision
        logger.warning(
            f"Your system has {total_ram:.2f} GB of RAM. This is below our reccomendation of 32GB for this type of training. Using half precision."
        )
        model = model.to(dev).half()  # Convert model to float16
    else:
        model = model.to(dev)

    # adafactor and gradient checkpointing are memory friendly, we opt to use these in the CPU/MPS loop to fit 7b models.
    optimizer = Adafactor(
        model.parameters(), lr=2e-5, scale_parameter=False, relative_step=False
    )
    model.gradient_checkpointing_enable()

    model.train()

    # For each epoch do the following:
    # 1. Incremement Loading Bar
    # 2. put zeros onto the proper device so we can store some values we need
    # 3. put each part of the batch onto the device as well using fp6
    # 4. Forward pass on the model. If using half, we need to do some special handling. If not, then just put the batch onto the device using fp32
    # 5. Calculate the loss
    # 6. backward pass on the loss
    # 7. We are using gradient accumulation to be memory efficient, so if this step is divisible by 4, take a step and zero out the gradients (freeing memory)
    # 8. clear out some caches if on MPS
    # 9. Incrememnt loading bar
    # 10. at the end og the epoch, save the checpoint to a samples folder and convert to GGUF as well
    for epoch in range(train_args.num_epochs):
        dataloader.batch_sampler.set_epoch(epoch)
        inner_pb = tqdm(range(len(dataloader)), desc=f"Epoch {epoch}")
        aggregated_values = torch.zeros(3, dtype=torch.float16).to(dev)

        for step, batch in enumerate(dataloader):
            aggregated_values[0] = batch.pop("num_loss_counted_tokens")
            aggregated_values[1] = len(batch["input_ids"])

            # Move and cast batch data to device
            for k in batch:
                if total_ram < 16:
                    if k in ["input_ids", "attention_mask"]:
                        # these two need to be of type long if using .half()
                        batch[k] = batch[k].to(device=dev, dtype=torch.long)
                    else:
                        batch[k] = batch[k].to(device=dev, dtype=torch.float16)
                else:
                    batch[k] = batch[k].to(device=dev)

            output = model(**batch, use_cache=False)
            loss = output.loss
            aggregated_values[2] = loss.item()

            num_loss_counted_tokens = aggregated_values[0]
            loss = loss / num_loss_counted_tokens

            loss = loss / accum  # Scale the loss for accumulation steps

            logger.info(f"\nEpoch: {epoch}, Step: {step + 1}, Rank: 0, loss = {loss}")

            # Gradient accumulation
            loss.backward()  # Backward pass

            # Clear cache before optimizer step
            # below we clear the MPS cache quite a bit. MPS caching is different from CPU.
            # on MPS systems with 32 or 64 GB of unified memory, it is important to clear the cache beacuse often we are paging a lot of the data that can't fit directly on GPU
            # CPU doesn't have an equivalent cache clearing method, only GPU devices like MPS, CUDA, ROCM, etc.
            if dev.type == "mps":
                torch.mps.empty_cache()

            # if we are on a step which is divisible by 4, step and zero gradients
            if (step + 1) % accum == 0:
                optimizer.step()  # Optimizer step
                optimizer.zero_grad()  # Zero gradients

                # Clear cache after optimizer step
                if dev.type == "mps":
                    torch.mps.empty_cache()

            inner_pb.update(1)

        # Clear cache at the end of the epoch if needed
        if dev.type == "mps":
            torch.mps.empty_cache()

        output_dir = (
            Path(train_args.ckpt_output_dir) / "hf_format" / f"samples_{(epoch*8)}"
        )

        logger.info(f"Saving Model to: {output_dir}")
        model_state = model.state_dict()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_model_file = output_dir / "pytorch_model.bin"
        output_config_file = output_dir / "config.json"

        tmp_conf = copy(model.config)

        torch.save(model_state, str(output_model_file))
        tmp_conf.to_json_file(str(output_config_file))
        tokenizer.save_pretrained(str(output_dir))

        # convert to GGUF at the end so llamacpp can serve the model originally in .bin
        llamacpp_convert_to_gguf.convert_llama_to_gguf(
            model=output_dir,
            pad_vocab=True,
            skip_unknown=True,
            outfile=os.path.join(output_dir, "pytorch_model.gguf"),
        )

        # quantize the model so everyone can run it
        gguf_model_dir = os.path.join(output_dir, "pytorch_model.gguf")
        gguf_model_q_dir = os.path.join(output_dir, "pytorch_model-Q4_K_M.gguf")
        run_quantize(gguf_model_dir, gguf_model_q_dir, "Q4_K_M")


def pad_collate_fn(batch, pad_token_id):
    lens = np.array([len(item["input_ids"]) for item in batch])
    max_len = max(lens)
    # Third Party
    import torch
    import torch.nn.functional as F

    input_ids = torch.stack(
        [
            F.pad(
                item["input_ids"],
                (max_len - len(item["input_ids"]), 0),
                mode="constant",
                value=pad_token_id,
            )
            for item in batch
        ]
    )
    labels = torch.stack(
        [
            F.pad(
                item["labels"],
                (max_len - len(item["labels"]), 0),
                mode="constant",
                value=-100,
            )
            for item in batch
        ]
    )
    num_loss_counted_tokens = (labels != -100).sum()

    attention_mask = torch.stack(
        [
            F.pad(
                item["attention_mask"],
                (max_len - len(item["attention_mask"]), 0),
                mode="constant",
                value=0,
            )
            for item in batch
        ]
    )
    logger.info(
        f"\033[96m total tokens: {max_len * len(batch)} num samples: {len(batch)} num padding tokens: {max_len * len(batch) - lens.sum()} - rank: {0} "
        f"max len: {max_len} min len: {min(lens)} avg len: {lens.mean()} "
        f"num_loss_counted_tokens: {num_loss_counted_tokens}\033[0m"
    )

    return {
        "input_ids": input_ids,
        "labels": labels,
        "num_loss_counted_tokens": num_loss_counted_tokens,
        "attention_mask": attention_mask,
    }
