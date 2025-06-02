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
from transformers import Adafactor, AutoConfig, AutoModelForCausalLM
import numpy as np
import psutil

# First Party
from instructlab.llamacpp import llamacpp_convert_to_gguf
from instructlab.utils import is_macos_with_m_chip

logger = logging.getLogger(__name__)


def convert_loss_to_reduce_sum(model):
    """
    this is necessary because multipack changes the samples per gpu, which biases the gradients to be larger for batches with less samples but longer lengths.
    """
    # Standard
    from typing import List, Optional

    # Third Party
    import torch

    def reduce_sum_forward(
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        # pylint: disable=unused-argument
        **deprecated_arguments,
    ):
        output = model.__original_forward__(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
        )

        logits = None
        loss = None
        return_dict = isinstance(output, dict)
        if return_dict:
            logits = output.logits
        else:
            # just checks that the output from the model is in the shape we expect,
            # and that one of the tuple elements is the loss and one is the logits
            if not (
                len(output) == 2
                and (
                    (len(output[0].shape) == 3 and len(output[1].shape) == 0)
                    or (len(output[1].shape) == 3 and len(output[0].shape) == 0)
                )
            ):
                raise ValueError(
                    "Output does not match the expected structure. "
                    "Expected a tuple of length 2 with one element having shape of rank 3 and the other of rank 0."
                )
            logits = output[0] if len(output[0].shape) == 3 else output[1]

        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Ensure tensors are on the same device
            shift_labels = shift_labels.to(shift_logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            return ((loss,) + output) if loss is not None else output

        output.loss = loss
        return output

    model.__original_forward__ = model.forward
    model.forward = reduce_sum_forward
    return model


def setup_model(
    train_args,
    tokenizer,
    dataset,
    optimize_memory: bool,
    packing_max_batch_len: int,
    accum: int,
):
    # pylint: disable=no-name-in-module
    # Third Party
    from instructlab.training import multipack_sampler
    from torch.utils.data import DataLoader

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

    # if the user specified --optimize-memory OR they are on a Mac, set dtype=auto
    torch_dtype = "auto" if (optimize_memory or is_macos_with_m_chip()) else "float32"
    # auto config based on model path
    config = AutoConfig.from_pretrained(
        train_args.model_path, torchscript=True, trust_remote_code=True
    )
    # auto model based on model path
    model = AutoModelForCausalLM.from_pretrained(
        train_args.model_path,
        torch_dtype=torch_dtype,
        quantization_config=None,
        config=config,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
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

    # ensure the model has any tokens which were added to the tokenizer
    if tokenizer.pad_token_id is not None and model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.bos_token_id is not None and model.config.bos_token_id is None:
        model.config.bos_token_id = tokenizer.bos_token_id
    if tokenizer.eos_token_id is not None and model.config.eos_token_id is None:
        model.config.eos_token_id = tokenizer.eos_token_id

    model = convert_loss_to_reduce_sum(model)
    return model, dataloader


def train(train_args, device, optimize_memory):
    """
    train runs a CPU and MacOS optimized version of full fine tuning.
    Adafactor is the optimizer of choice and the multiprocessing method is set to spawn.
    Dataloading functions imported from the training library.
    """

    # pylint: disable=no-name-in-module
    # Third Party
    from instructlab.training import config
    from instructlab.training import data_process as dp
    from instructlab.training import multipack_sampler, token_dataset, tokenizer_utils
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
    tokenizer = tokenizer_utils.setup_tokenizer(
        train_args.model_path, train_args.chat_tmpl_path
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
            is_padding=False,
            dataset=dataset,
            seed=47,
        )
    )

    model, dataloader = setup_model(
        train_args, tokenizer, dataset, optimize_memory, packing_max_batch_len, accum
    )

    # Get virtual memory statistics
    memory_info = psutil.virtual_memory()

    # set device based off argument given
    dev = torch.device(device)

    # Total RAM
    total_ram = memory_info.total / (1024**3)  # Convert to GB
    logger.info(f"Total RAM: {total_ram:.2f} GB")
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
        aggregated_values = torch.zeros(3, dtype=torch.float32).to(dev)

        # in order to correctly calculate the loss, we need to divide each microbatch by
        # a constant factor, so that we can later correct it by the actual `total_minibatch_tokens` amount
        total_minibatch_tokens = 0
        interim_batch_denominator = packing_max_batch_len * accum
        loss_accum = 0.0  # track this for logging puproses

        for step, batch in enumerate(dataloader):
            aggregated_values[0] = batch.pop("num_loss_counted_tokens")
            aggregated_values[1] = len(batch["input_ids"])

            # Move and cast batch data to device
            for k in batch:
                batch[k] = batch[k].to(device=dev)

            output = model(**batch, use_cache=False, return_dict=False)
            loss = None
            if isinstance(output, tuple):
                loss = output[0]
                if len(output[0].shape) != 0:
                    raise ValueError(
                        "When output is a tuple, the loss should be the first element"
                    )
            else:
                loss = output.loss
            if loss is None:
                raise ValueError(
                    "Loss is None. Ensure the model's output contains a valid loss."
                )

            aggregated_values[2] = loss.item()

            num_loss_counted_tokens = aggregated_values[0]
            total_minibatch_tokens += num_loss_counted_tokens

            # here we need to correctly rescale the loss, so we divide by the packing_max_batch_len
            # in order to overshoot the average, and then we will later multiply each gradient
            # by a correction term
            loss_orig = loss.detach().cpu().item()
            loss_accum += loss
            loss = loss / interim_batch_denominator

            per_batch_loss = loss_orig / num_loss_counted_tokens

            logger.info(
                f"\nEpoch: {epoch}, Step: {step + 1}, Loss per batch: {loss.detach().item()}, Actual Loss Per Batch = {per_batch_loss}, accumulated loss: {loss_accum.item()}"
            )

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
                # lets correct all of the gradients
                for param in model.parameters():
                    grad = param.grad
                    assert grad is not None
                    correction_term = interim_batch_denominator / total_minibatch_tokens
                    param.grad *= correction_term

                optimizer.step()  # Optimizer step
                optimizer.zero_grad()  # Zero gradients

                # reset all of the accumulated data
                total_minibatch_tokens = 0.0
                loss_accum = 0.0

                # Clear cache after optimizer step
                if dev.type == "mps":
                    torch.mps.empty_cache()

            inner_pb.update(1)

        # Clear cache at the end of the epoch if needed
        if dev.type == "mps":
            torch.mps.empty_cache()

        output_dir = (
            Path(train_args.ckpt_output_dir) / "hf_format" / f"samples_{(epoch * 8)}"
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
        f"\033[96m total tokens: {max_len * len(batch)} num samples: {len(batch)} num padding tokens: {max_len * len(batch) - lens.sum()}"
        f"max len: {max_len} min len: {min(lens)} avg len: {lens.mean()} "
        f"num_loss_counted_tokens: {num_loss_counted_tokens}\033[0m"
    )

    return {
        "input_ids": input_ids,
        "labels": labels,
        "num_loss_counted_tokens": num_loss_counted_tokens,
        "attention_mask": attention_mask,
    }
