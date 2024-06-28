# SPDX-License-Identifier: MIT
# Copyright © 2023 Apple Inc.

# Standard
from pathlib import Path
from typing import Optional
import json
import math
import time

# Third Party
from mlx.utils import tree_flatten
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

# Local
from . import utils as lora_utils
from .models.lora import LoRALinear


class Dataset:
    """
    Light-weight wrapper to hold lines from a jsonl file
    """

    def __init__(self, path: Path, key: str = "text"):
        if not path.exists():
            self._data = None
        else:
            with open(path, "r") as fid:
                self._data = [json.loads(l) for l in fid]
        self._key = key

    def __getitem__(self, idx: int):
        return self._data[idx][self._key]

    def __len__(self):
        return len(self._data)


def load(data, train, test):
    def load_and_check(name):
        dataset_path = Path(data) / f"{name}.jsonl"
        try:
            return Dataset(dataset_path)
        except Exception as e:
            print(f"Unable to build dataset {dataset_path} ({e})")
            raise

    names = ("train", "valid", "test")
    train_dataset_path, valid_dataset_path, test_dataset_path = (
        load_and_check(n) for n in names
    )

    if train and len(train_dataset_path) == 0:
        raise ValueError(
            "Training set not found or empty. Must provide training set for fine-tuning."
        )
    if train and len(valid_dataset_path) == 0:
        raise ValueError(
            "Validation set not found or empty. Must provide validation set for fine-tuning."
        )
    if test and len(test_dataset_path) == 0:
        raise ValueError(
            "Test set not found or empty. Must provide test set for evaluation."
        )
    return train_dataset_path, valid_dataset_path, test_dataset_path


def loss(model, inputs, targets, lengths):
    # Run model on inputs
    logits, _ = model(inputs)
    logits = logits.astype(mx.float32)

    # Mask padding tokens
    length_mask = mx.arange(inputs.shape[1])[None, :] < lengths[:, None]

    # Calculate the loss
    ce = nn.losses.cross_entropy(logits, targets) * length_mask
    ntoks = length_mask.sum()
    ce = ce.sum() / ntoks
    return ce, ntoks


def iterate_batches(dset, tokenizer, batch_size, train=False):
    # Shuffle indices
    while True:
        indices = np.arange(len(dset))
        if train:
            indices = np.random.permutation(indices)

        # Collect batches from dataset
        for i in range(0, len(indices) - batch_size + 1, batch_size):
            # Encode batch
            batch = [tokenizer.encode(dset[indices[i + j]]) for j in range(batch_size)]
            lengths = [len(x) for x in batch]

            # Check if any sequence is longer than 2048 tokens
            if max(lengths) > 2048:
                print(
                    "[WARNING] Some sequences are longer than 2048 tokens. "
                    "Consider pre-splitting your data to save memory."
                )

            # Pad to the max length
            batch_arr = np.zeros((batch_size, max(lengths)), np.int32)

            for j in range(batch_size):
                batch_arr[j, : lengths[j]] = batch[j]
            batch = mx.array(batch_arr)
            yield batch[:, :-1], batch[:, 1:], mx.array(lengths)

        if not train:
            break


def evaluate(model, dataset, loss, tokenizer, batch_size, num_batches):
    all_losses = []
    ntokens = 0
    for it, batch in zip(
        range(num_batches),
        iterate_batches(dataset, tokenizer, batch_size),
    ):
        losses, toks = loss(model, *batch)
        all_losses.append((losses * toks).item())
        ntokens += toks.item()

    return np.sum(all_losses) / ntokens


def train_model(
    model,
    train_set,
    val_set,
    optimizer,
    loss,
    tokenizer,
    iters,
    batch_size,
    steps_per_report,
    steps_per_eval,
    val_batches,
    save_every,
    adapter_file,
):
    # Create value and grad function for loss
    loss_value_and_grad = nn.value_and_grad(model, loss)

    losses = []
    n_tokens = 0

    # Main training loop
    start = time.perf_counter()
    for it, batch in zip(
        range(iters),
        iterate_batches(train_set, tokenizer, batch_size, train=True),
    ):
        # Forward and backward pass
        (lvalue, toks), grad = loss_value_and_grad(model, *batch)

        # Model update
        optimizer.update(model, grad)
        mx.eval(model.parameters(), optimizer.state, lvalue)

        # Record loss
        losses.append(lvalue.item())
        n_tokens += toks.item()

        # Report training loss if needed
        if (it + 1) % steps_per_report == 0:
            train_loss = np.mean(losses)

            stop = time.perf_counter()
            print(
                f"Iter {it+1:03d}: Train loss {train_loss:.3f}, "
                f"It/sec {steps_per_report / (stop - start):.3f}, "
                f"Tokens/sec {float(n_tokens) / (stop - start):.3f}"
            )
            losses = []
            n_tokens = 0
            start = time.perf_counter()

        # Report validation loss if needed
        if it == 0 or (it + 1) % steps_per_eval == 0:
            stop = time.perf_counter()
            val_loss = evaluate(
                model, val_set, loss, tokenizer, batch_size, val_batches
            )
            epoch = (it * batch_size) // len(train_set)
            print(
                f"Epoch {epoch + 1}: "
                f"Iter {it + 1}: "
                f"Val loss {val_loss:.3f}, "
                f"Val took {(time.perf_counter() - stop):.3f}s"
            )

            start = time.perf_counter()

        # Save adapter weights if needed
        if (it + 1) % save_every == 0:
            mx.savez(adapter_file, **dict(tree_flatten(model.trainable_parameters())))
            a, b = adapter_file.split(".")
            fn = f"{a}-{it+1:03d}.{b}"
            mx.savez(fn, **dict(tree_flatten(model.trainable_parameters())))
            print(f"Iter {it + 1}: Saved adapter weights to {fn}.")


def generate(model, prompt, tokenizer, stream, temp, max_tokens):
    if stream:
        print(prompt, end="", flush=True)

    prompt = mx.array(tokenizer.encode(prompt))

    tokens = []
    skip = 0
    for token, n in zip(
        lora_utils.generate(prompt, model, temp),
        range(max_tokens),
    ):
        if token == tokenizer.eos_token_id:
            break

        tokens.append(token.item())
        s = tokenizer.decode(tokens)
        if len(s) - skip > 1:
            if stream:
                print(s[skip:-1], end="", flush=True)
            skip = len(s) - 1
    if stream:
        print(tokenizer.decode(tokens)[skip:], flush=True)
    else:
        print("=" * 10)
        print(tokenizer.decode(tokens).strip())
        print("=" * 10)
    if len(tokens) == 0:
        print("No tokens generated for this prompt")
        return


def load_and_train(
    model: str = "mlx_model",
    max_tokens: int = 100,
    temp: float = 0.8,
    prompt: Optional[str] = None,
    train: bool = False,
    data: str = "data/",
    lora_layers: int = 16,
    batch_size: int = 4,
    iters: int = 1000,
    val_batches: int = 25,
    learning_rate: float = 1e-5,
    steps_per_report: int = 10,
    steps_per_eval: int = 200,
    resume_adapter_file: Optional[str] = None,
    adapter_file: str = "adapters.npz",
    save_every: int = 100,
    test: bool = False,
    stream: bool = False,
    no_adapter: bool = False,
    test_batches: int = 500,
    seed: int = 0,
):
    """LoRA or QLoRA fine tuning."""
    np.random.seed(seed)

    print("Loading pretrained model")
    model, tokenizer, _ = lora_utils.load(model)

    # Freeze all layers other than LORA linears
    model.freeze()
    if not no_adapter:
        for l in model.model.layers[len(model.model.layers) - lora_layers :]:
            l.self_attn.q_proj = LoRALinear.from_linear(l.self_attn.q_proj)
            l.self_attn.v_proj = LoRALinear.from_linear(l.self_attn.v_proj)
            if hasattr(l, "block_sparse_moe"):
                l.block_sparse_moe.gate = LoRALinear.from_linear(
                    l.block_sparse_moe.gate
                )
    else:
        print("LoRA init skipped")

    p = sum(v.size for _, v in tree_flatten(model.parameters())) / 10**6
    print(f"Total parameters {p:.3f}M")
    p = sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 10**6
    print(f"Trainable parameters {p:.3f}M")

    print("Loading datasets")
    train_set, valid_set, test_set = load(data, train, test)

    # Resume training the given adapters.
    if resume_adapter_file is not None:
        print(f"Loading pretrained adapters from {resume_adapter_file}")
        model.load_weights(resume_adapter_file, strict=False)

    if train:
        print("*********")
        print("")
        print("ᕙ(•̀‸•́‶)ᕗ  Training has started! ᕙ(•̀‸•́‶)ᕗ ")
        print("")
        print("*********")
        opt = optim.Adam(learning_rate=learning_rate)

        # Train model
        train_model(
            model,
            train_set,
            valid_set,
            opt,
            loss,
            tokenizer,
            iters,
            batch_size,
            steps_per_report,
            steps_per_eval,
            val_batches,
            save_every,
            adapter_file,
        )

        # Save adapter weights
        mx.savez(adapter_file, **dict(tree_flatten(model.trainable_parameters())))

    if not no_adapter:
        # Load the LoRA adapter weights which we assume should exist by this point
        if not Path(adapter_file).is_file():
            raise ValueError(
                f"Adapter file {adapter_file} missing. "
                "Use --train to learn and save the adapters.npz."
            )
        model.load_weights(adapter_file, strict=False)
    else:
        print("LoRA loading skipped")

    if test:
        print("Testing")
        model.eval()
        test_loss = evaluate(
            model,
            test_set,
            loss,
            tokenizer,
            batch_size,
            num_batches=test_batches,
        )
        test_ppl = math.exp(test_loss)

        print(f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}.")

    if prompt is not None:
        print("Generating")
        generate(model, prompt, tokenizer, stream, temp, max_tokens)
