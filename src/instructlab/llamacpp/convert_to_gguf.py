#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Future
from __future__ import annotations

# Standard
from enum import IntEnum
from hashlib import sha256
from itertools import chain
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ContextManager,
    Iterable,
    Iterator,
    Literal,
    Sequence,
    TypeVar,
    cast,
)
import contextlib
import json
import logging
import math
import os
import sys

# Third Party
import numpy as np
import torch

if TYPE_CHECKING:
    # Third Party
    from torch import Tensor

if "NO_LOCAL_GGUF" not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / "gguf-py"))
# Third Party
import gguf

logger = logging.getLogger("hf-to-gguf")


###### MODEL DEFINITIONS ######


class SentencePieceTokenTypes(IntEnum):
    NORMAL = 1
    UNKNOWN = 2
    CONTROL = 3
    USER_DEFINED = 4
    UNUSED = 5
    BYTE = 6


AnyModel = TypeVar("AnyModel", bound="type[Model]")


class Model:
    _model_classes: dict[str, type[Model]] = {}

    dir_model: Path
    ftype: gguf.LlamaFileType
    fname_out: Path
    is_big_endian: bool
    endianess: gguf.GGUFEndian
    use_temp_file: bool
    lazy: bool
    part_names: list[str]
    is_safetensors: bool
    hparams: dict[str, Any]
    block_count: int
    tensor_map: gguf.TensorNameMap
    tensor_names: set[str] | None
    gguf_writer: gguf.GGUFWriter
    model_name: str | None
    metadata_override: Path | None
    dir_model_card: Path

    # subclasses should define this!
    model_arch: gguf.MODEL_ARCH

    def __init__(
        self,
        dir_model: Path,
        ftype: gguf.LlamaFileType,
        fname_out: Path,
        is_big_endian: bool = False,
        use_temp_file: bool = False,
        eager: bool = False,
        metadata_override: Path | None = None,
        model_name: str | None = None,
        split_max_tensors: int = 0,
        split_max_size: int = 0,
        dry_run: bool = False,
        small_first_shard: bool = False,
        hparams: dict[str, Any] | None = None,
    ):
        if type(self) is Model:
            raise TypeError(
                f"{type(self).__name__!r} should not be directly instantiated"
            )

        self.dir_model = dir_model
        self.ftype = ftype
        self.fname_out = fname_out
        self.is_big_endian = is_big_endian
        self.endianess = (
            gguf.GGUFEndian.BIG if is_big_endian else gguf.GGUFEndian.LITTLE
        )
        self.use_temp_file = use_temp_file
        self.lazy = not eager
        self.part_names = Model.get_model_part_names(
            self.dir_model, "model", ".safetensors"
        )
        self.is_safetensors = len(self.part_names) > 0
        if not self.is_safetensors:
            self.part_names = Model.get_model_part_names(
                self.dir_model, "pytorch_model", ".bin"
            )
        self.hparams = (
            Model.load_hparams(self.dir_model) if hparams is None else hparams
        )
        self.block_count = self.find_hparam(
            ["n_layers", "num_hidden_layers", "n_layer", "num_layers"]
        )
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)
        self.tensor_names = None
        self.metadata_override = metadata_override
        self.model_name = model_name
        self.dir_model_card = dir_model  # overridden in convert_lora_to_gguf.py

        # Apply heuristics to figure out typical tensor encoding based on first layer tensor encoding type
        if self.ftype == gguf.LlamaFileType.GUESSED:
            # NOTE: can't use field "torch_dtype" in config.json, because some finetunes lie.
            _, first_tensor = next(self.get_tensors())
            if first_tensor.dtype == torch.float16:
                logger.info(
                    f"choosing --outtype f16 from first tensor type ({first_tensor.dtype})"
                )
                self.ftype = gguf.LlamaFileType.MOSTLY_F16
            else:
                logger.info(
                    f"choosing --outtype bf16 from first tensor type ({first_tensor.dtype})"
                )
                self.ftype = gguf.LlamaFileType.MOSTLY_BF16

        # Configure GGUF Writer
        self.gguf_writer = gguf.GGUFWriter(
            path=None,
            arch=gguf.MODEL_ARCH_NAMES[self.model_arch],
            endianess=self.endianess,
            use_temp_file=self.use_temp_file,
            split_max_tensors=split_max_tensors,
            split_max_size=split_max_size,
            dry_run=dry_run,
            small_first_shard=small_first_shard,
        )

    @classmethod
    def __init_subclass__(cls):
        # can't use an abstract property, because overriding it without type errors
        # would require using decorated functions instead of simply defining the property
        if "model_arch" not in cls.__dict__:
            raise TypeError(f"Missing property 'model_arch' for {cls.__name__!r}")

    def find_hparam(self, keys: Iterable[str], optional: bool = False) -> Any:
        key = next((k for k in keys if k in self.hparams), None)
        if key is not None:
            return self.hparams[key]
        if optional:
            return None
        raise KeyError(f"could not find any of: {keys}")

    def set_vocab(self):
        self._set_vocab_gpt2()

    def get_tensors(self) -> Iterator[tuple[str, Tensor]]:
        tensor_names_from_parts: set[str] = set()

        index_name = "model.safetensors" if self.is_safetensors else "pytorch_model.bin"
        index_name += ".index.json"
        index_file = self.dir_model / index_name

        if index_file.is_file():
            self.tensor_names = set()
            logger.info(f"gguf: loading model weight map from '{index_name}'")
            with open(index_file, "r", encoding="utf-8") as f:
                index: dict[str, Any] = json.load(f)
                weight_map = index.get("weight_map")
                if weight_map is None or not isinstance(weight_map, dict):
                    raise ValueError(f"Can't load 'weight_map' from {index_name!r}")
                self.tensor_names.update(weight_map.keys())
        else:
            self.tensor_names = tensor_names_from_parts
            weight_map = {}

        for part_name in self.part_names:
            logger.info(f"gguf: loading model part '{part_name}'")
            ctx: ContextManager[Any]
            if self.is_safetensors:
                # Third Party
                from safetensors import safe_open

                ctx = cast(
                    ContextManager[Any],
                    safe_open(self.dir_model / part_name, framework="pt", device="cpu"),
                )
            else:
                ctx = contextlib.nullcontext(
                    torch.load(
                        str(self.dir_model / part_name),
                        map_location="cpu",
                        mmap=True,
                        weights_only=True,
                    )
                )

            with ctx as model_part:
                tensor_names_from_parts.update(model_part.keys())

                for name in model_part.keys():
                    if self.is_safetensors:
                        if self.lazy:
                            data = model_part.get_slice(name)
                            data = LazyTorchTensor.from_safetensors_slice(data)
                        else:
                            data = model_part.get_tensor(name)
                    else:
                        data = model_part[name]
                        if self.lazy:
                            data = LazyTorchTensor.from_eager(data)
                    yield name, data

        # verify tensor name presence and identify potentially missing files
        if len(tensor_names_from_parts.symmetric_difference(self.tensor_names)) > 0:
            missing = sorted(self.tensor_names.difference(tensor_names_from_parts))
            extra = sorted(tensor_names_from_parts.difference(self.tensor_names))
            missing_files = sorted(
                set(weight_map[n] for n in missing if n in weight_map)
            )
            if len(extra) == 0 and len(missing_files) > 0:
                raise ValueError(f"Missing or incomplete model files: {missing_files}")
            else:
                raise ValueError(
                    "Mismatch between weight map and model parts for tensor names:\n"
                    f"Missing tensors: {missing}\n"
                    f"Extra tensors: {extra}"
                )

    def format_tensor_name(
        self, key: gguf.MODEL_TENSOR, bid: int | None = None, suffix: str = ".weight"
    ) -> str:
        if key not in gguf.MODEL_TENSORS[self.model_arch]:
            raise ValueError(
                f"Missing {key!r} for MODEL_TENSORS of {self.model_arch!r}"
            )
        name: str = gguf.TENSOR_NAMES[key]
        if "{bid}" in name:
            assert bid is not None
            name = name.format(bid=bid)
        return name + suffix

    def match_model_tensor_name(
        self,
        name: str,
        key: gguf.MODEL_TENSOR,
        bid: int | None,
        suffix: str = ".weight",
    ) -> bool:
        if key not in gguf.MODEL_TENSORS[self.model_arch]:
            return False
        key_name: str = gguf.TENSOR_NAMES[key]
        if "{bid}" in key_name:
            if bid is None:
                return False
            key_name = key_name.format(bid=bid)
        else:
            if bid is not None:
                return False
        return name == (key_name + suffix)

    def map_tensor_name(
        self, name: str, try_suffixes: Sequence[str] = (".weight", ".bias")
    ) -> str:
        new_name = self.tensor_map.get_name(key=name, try_suffixes=try_suffixes)
        if new_name is None:
            raise ValueError(f"Can not map tensor {name!r}")
        return new_name

    def set_gguf_parameters(self):
        self.gguf_writer.add_block_count(self.block_count)

        if (
            n_ctx := self.find_hparam(
                ["max_position_embeddings", "n_ctx"], optional=True
            )
        ) is not None:
            self.gguf_writer.add_context_length(n_ctx)
            logger.info(f"gguf: context length = {n_ctx}")

        n_embd = self.find_hparam(["hidden_size", "n_embd"])
        self.gguf_writer.add_embedding_length(n_embd)
        logger.info(f"gguf: embedding length = {n_embd}")

        if (
            n_ff := self.find_hparam(["intermediate_size", "n_inner"], optional=True)
        ) is not None:
            self.gguf_writer.add_feed_forward_length(n_ff)
            logger.info(f"gguf: feed forward length = {n_ff}")

        n_head = self.find_hparam(["num_attention_heads", "n_head"])
        self.gguf_writer.add_head_count(n_head)
        logger.info(f"gguf: head count = {n_head}")

        if (n_head_kv := self.hparams.get("num_key_value_heads")) is not None:
            self.gguf_writer.add_head_count_kv(n_head_kv)
            logger.info(f"gguf: key-value head count = {n_head_kv}")

        if (rope_theta := self.hparams.get("rope_theta")) is not None:
            self.gguf_writer.add_rope_freq_base(rope_theta)
            logger.info(f"gguf: rope theta = {rope_theta}")
        if (f_rms_eps := self.hparams.get("rms_norm_eps")) is not None:
            self.gguf_writer.add_layer_norm_rms_eps(f_rms_eps)
            logger.info(f"gguf: rms norm epsilon = {f_rms_eps}")
        if (
            f_norm_eps := self.find_hparam(
                ["layer_norm_eps", "layer_norm_epsilon", "norm_epsilon"], optional=True
            )
        ) is not None:
            self.gguf_writer.add_layer_norm_eps(f_norm_eps)
            logger.info(f"gguf: layer norm epsilon = {f_norm_eps}")
        if (n_experts := self.hparams.get("num_local_experts")) is not None:
            self.gguf_writer.add_expert_count(n_experts)
            logger.info(f"gguf: expert count = {n_experts}")
        if (n_experts_used := self.hparams.get("num_experts_per_tok")) is not None:
            self.gguf_writer.add_expert_used_count(n_experts_used)
            logger.info(f"gguf: experts used count = {n_experts_used}")

        if (head_dim := self.hparams.get("head_dim")) is not None:
            self.gguf_writer.add_key_length(head_dim)
            self.gguf_writer.add_value_length(head_dim)

        self.gguf_writer.add_file_type(self.ftype)
        logger.info(f"gguf: file type = {self.ftype}")

    def modify_tensors(
        self, data_torch: Tensor, name: str, bid: int | None
    ) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused

        return [(self.map_tensor_name(name), data_torch)]

    def tensor_force_quant(
        self, name: str, new_name: str, bid: int | None, n_dims: int
    ) -> gguf.GGMLQuantizationType | bool:
        del name, new_name, bid, n_dims  # unused

        return False

    # some models need extra generated tensors (like rope_freqs)
    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        return ()

    def prepare_tensors(self):
        max_name_len = max(len(s) for _, s in self.tensor_map.mapping.values()) + len(
            ".weight,"
        )

        for name, data_torch in chain(
            self.generate_extra_tensors(), self.get_tensors()
        ):
            # we don't need these
            if name.endswith(
                (".attention.masked_bias", ".attention.bias", ".rotary_emb.inv_freq")
            ):
                continue

            old_dtype = data_torch.dtype

            # convert any unsupported data types to float32
            if data_torch.dtype not in (torch.float16, torch.float32):
                data_torch = data_torch.to(torch.float32)

            # use the first number-like part of the tensor name as the block id
            bid = None
            for part in name.split("."):
                if part.isdecimal():
                    bid = int(part)
                    break

            for new_name, data_torch in self.modify_tensors(data_torch, name, bid):
                data = data_torch.squeeze().numpy()

                # if data ends up empty, it means data_torch was a scalar tensor -> restore
                if len(data.shape) == 0:
                    data = data_torch.numpy()

                n_dims = len(data.shape)
                data_qtype: gguf.GGMLQuantizationType | bool = self.tensor_force_quant(
                    name, new_name, bid, n_dims
                )

                # Most of the codebase that takes in 1D tensors or norms only handles F32 tensors
                if n_dims <= 1 or new_name.endswith("_norm.weight"):
                    data_qtype = gguf.GGMLQuantizationType.F32

                # Conditions should closely match those in llama_model_quantize_internal in llama.cpp
                # Some tensor types are always in float32
                if data_qtype is False and (
                    any(
                        self.match_model_tensor_name(new_name, key, bid)
                        for key in (
                            gguf.MODEL_TENSOR.FFN_GATE_INP,
                            gguf.MODEL_TENSOR.POS_EMBD,
                            gguf.MODEL_TENSOR.TOKEN_TYPES,
                            gguf.MODEL_TENSOR.SSM_CONV1D,
                            gguf.MODEL_TENSOR.TIME_MIX_FIRST,
                            gguf.MODEL_TENSOR.TIME_MIX_W1,
                            gguf.MODEL_TENSOR.TIME_MIX_W2,
                            gguf.MODEL_TENSOR.TIME_MIX_DECAY_W1,
                            gguf.MODEL_TENSOR.TIME_MIX_DECAY_W2,
                        )
                    )
                    or not new_name.endswith(".weight")
                ):
                    data_qtype = gguf.GGMLQuantizationType.F32

                if data_qtype is False and any(
                    self.match_model_tensor_name(new_name, key, bid)
                    for key in (
                        gguf.MODEL_TENSOR.TOKEN_EMBD,
                        gguf.MODEL_TENSOR.OUTPUT,
                    )
                ):
                    if self.ftype in (
                        gguf.LlamaFileType.MOSTLY_TQ1_0,
                        gguf.LlamaFileType.MOSTLY_TQ2_0,
                    ):
                        # TODO: use Q4_K and Q6_K
                        data_qtype = gguf.GGMLQuantizationType.F16

                # No override (data_qtype is False), or wants to be quantized (data_qtype is True)
                if isinstance(data_qtype, bool):
                    if self.ftype == gguf.LlamaFileType.ALL_F32:
                        data_qtype = gguf.GGMLQuantizationType.F32
                    elif self.ftype == gguf.LlamaFileType.MOSTLY_F16:
                        data_qtype = gguf.GGMLQuantizationType.F16
                    elif self.ftype == gguf.LlamaFileType.MOSTLY_BF16:
                        data_qtype = gguf.GGMLQuantizationType.BF16
                    elif self.ftype == gguf.LlamaFileType.MOSTLY_Q8_0:
                        data_qtype = gguf.GGMLQuantizationType.Q8_0
                    elif self.ftype == gguf.LlamaFileType.MOSTLY_TQ1_0:
                        data_qtype = gguf.GGMLQuantizationType.TQ1_0
                    elif self.ftype == gguf.LlamaFileType.MOSTLY_TQ2_0:
                        data_qtype = gguf.GGMLQuantizationType.TQ2_0
                    else:
                        raise ValueError(f"Unknown file type: {self.ftype.name}")

                try:
                    data = gguf.quants.quantize(data, data_qtype)
                except gguf.QuantError as e:
                    logger.warning("%s, %s", e, "falling back to F16")
                    data_qtype = gguf.GGMLQuantizationType.F16
                    data = gguf.quants.quantize(data, data_qtype)

                shape = (
                    gguf.quant_shape_from_byte_shape(data.shape, data_qtype)
                    if data.dtype == np.uint8
                    else data.shape
                )

                # reverse shape to make it similar to the internal ggml dimension order
                shape_str = f"{{{', '.join(str(n) for n in reversed(shape))}}}"

                # n_dims is implicit in the shape
                logger.info(
                    f"{f'%-{max_name_len}s' % f'{new_name},'} {old_dtype} --> {data_qtype.name}, shape = {shape_str}"
                )

                self.gguf_writer.add_tensor(new_name, data, raw_dtype=data_qtype)

    def set_type(self):
        self.gguf_writer.add_type(gguf.GGUFType.MODEL)

    def prepare_metadata(self, vocab_only: bool):
        total_params, shared_params, expert_params, expert_count = (
            self.gguf_writer.get_total_parameter_count()
        )

        self.metadata = gguf.Metadata.load(
            self.metadata_override, self.dir_model_card, self.model_name, total_params
        )

        # Fallback to model directory name if metadata name is still missing
        if self.metadata.name is None:
            self.metadata.name = self.dir_model.name

        # Generate parameter weight class (useful for leader boards) if not yet determined
        if self.metadata.size_label is None and total_params > 0:
            self.metadata.size_label = gguf.size_label(
                total_params, shared_params, expert_params, expert_count
            )

        # Extract the encoding scheme from the file type name. e.g. 'gguf.LlamaFileType.MOSTLY_Q8_0' --> 'Q8_0'
        output_type: str = self.ftype.name.partition("_")[2]

        # Filename Output
        if self.fname_out.is_dir():
            # Generate default filename based on model specification and available metadata
            if not vocab_only:
                fname_default: str = gguf.naming_convention(
                    self.metadata.name,
                    self.metadata.basename,
                    self.metadata.finetune,
                    self.metadata.version,
                    self.metadata.size_label,
                    output_type,
                    model_type="LoRA" if total_params < 0 else None,
                )
            else:
                fname_default: str = gguf.naming_convention(
                    self.metadata.name,
                    self.metadata.basename,
                    self.metadata.finetune,
                    self.metadata.version,
                    size_label=None,
                    output_type=None,
                    model_type="vocab",
                )

            # Use the default filename
            self.fname_out = self.fname_out / f"{fname_default}.gguf"
        else:
            # Output path is a custom defined templated filename
            # Note: `not is_dir()` is used because `.is_file()` will not detect
            #       file template strings as it doesn't actually exist as a file

            # Process templated file name with the output ftype, useful with the "auto" ftype
            self.fname_out = self.fname_out.parent / gguf.fill_templated_filename(
                self.fname_out.name, output_type
            )

        self.set_type()

        logger.info("Set meta model")
        self.metadata.set_gguf_meta_model(self.gguf_writer)

        logger.info("Set model parameters")
        self.set_gguf_parameters()

        logger.info("Set model tokenizer")
        self.set_vocab()

        logger.info("Set model quantization version")
        self.gguf_writer.add_quantization_version(gguf.GGML_QUANT_VERSION)

    def write(self):
        self.prepare_tensors()
        self.prepare_metadata(vocab_only=False)
        self.gguf_writer.write_header_to_file(path=self.fname_out)
        self.gguf_writer.write_kv_data_to_file()
        self.gguf_writer.write_tensors_to_file(progress=True)
        self.gguf_writer.close()

    def write_vocab(self):
        if len(self.gguf_writer.tensors) != 1:
            raise ValueError("Splitting the vocabulary is not supported")

        self.prepare_metadata(vocab_only=True)
        self.gguf_writer.write_header_to_file(path=self.fname_out)
        self.gguf_writer.write_kv_data_to_file()
        self.gguf_writer.close()

    @staticmethod
    def get_model_part_names(dir_model: Path, prefix: str, suffix: str) -> list[str]:
        part_names: list[str] = []
        for filename in os.listdir(dir_model):
            if filename.startswith(prefix) and filename.endswith(suffix):
                part_names.append(filename)

        part_names.sort()

        return part_names

    @staticmethod
    def load_hparams(dir_model: Path):
        with open(dir_model / "config.json", "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def register(cls, *names: str) -> Callable[[AnyModel], AnyModel]:
        assert names

        def func(modelcls: AnyModel) -> AnyModel:
            for name in names:
                cls._model_classes[name] = modelcls
            return modelcls

        return func

    @classmethod
    def from_model_architecture(cls, arch: str) -> type[Model]:
        try:
            return cls._model_classes[arch]
        except KeyError:
            raise NotImplementedError(f"Architecture {arch!r} not supported!") from None

    def does_token_look_special(self, token: str | bytes) -> bool:
        if isinstance(token, (bytes, bytearray)):
            token_text = token.decode(encoding="utf-8")
        elif isinstance(token, memoryview):
            token_text = token.tobytes().decode(encoding="utf-8")
        else:
            token_text = token

        # Some models mark some added tokens which ought to be control tokens as not special.
        # (e.g. command-r, command-r-plus, deepseek-coder, gemma{,-2})
        seems_special = token_text in (
            "<pad>",  # deepseek-coder
            "<mask>",
            "<2mass>",
            "[@BOS@]",  # gemma{,-2}
        )

        seems_special = seems_special or (
            token_text.startswith("<|") and token_text.endswith("|>")
        )
        seems_special = seems_special or (
            token_text.startswith("<｜") and token_text.endswith("｜>")
        )  # deepseek-coder

        # TODO: should these be marked as UNUSED instead? (maybe not)
        seems_special = seems_special or (
            token_text.startswith("<unused") and token_text.endswith(">")
        )  # gemma{,-2}

        return seems_special

    # used for GPT-2 BPE and WordPiece vocabs
    def get_vocab_base(self) -> tuple[list[str], list[int], str]:
        tokens: list[str] = []
        toktypes: list[int] = []

        # Third Party
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.dir_model)
        vocab_size = self.hparams.get("vocab_size", len(tokenizer.vocab))
        assert max(tokenizer.vocab.values()) < vocab_size

        tokpre = self.get_vocab_base_pre(tokenizer)

        reverse_vocab = {
            id_: encoded_tok for encoded_tok, id_ in tokenizer.vocab.items()
        }
        added_vocab = tokenizer.get_added_vocab()

        for i in range(vocab_size):
            if i not in reverse_vocab:
                tokens.append(f"[PAD{i}]")
                toktypes.append(gguf.TokenType.UNUSED)
            else:
                token: str = reverse_vocab[i]
                if token in added_vocab:
                    if tokenizer.added_tokens_decoder[
                        i
                    ].special or self.does_token_look_special(token):
                        toktypes.append(gguf.TokenType.CONTROL)
                    else:
                        token = token.replace(
                            b"\xe2\x96\x81".decode("utf-8"), " "
                        )  # pre-normalize user-defined spaces
                        toktypes.append(gguf.TokenType.USER_DEFINED)
                else:
                    toktypes.append(gguf.TokenType.NORMAL)
                tokens.append(token)

        return tokens, toktypes, tokpre

    # NOTE: this function is generated by convert_hf_to_gguf_update.py
    #       do not modify it manually!
    # ref:  https://github.com/ggerganov/llama.cpp/pull/6920
    # Marker: Start get_vocab_base_pre
    def get_vocab_base_pre(self, tokenizer) -> str:
        # encoding this string and hashing the resulting tokens would (hopefully) give us a unique identifier that
        # is specific for the BPE pre-tokenizer used by the model
        # we will use this unique identifier to write a "tokenizer.ggml.pre" entry in the GGUF file which we can
        # use in llama.cpp to implement the same pre-tokenizer

        chktxt = "\n \n\n \n\n\n \t \t\t \t\n  \n   \n    \n     \n🚀 (normal) 😶\u200d🌫️ (multiple emojis concatenated) ✅ 🦙🦙 3 33 333 3333 33333 333333 3333333 33333333 3.3 3..3 3...3 កាន់តែពិសេសអាច😁 ?我想在apple工作1314151天～ ------======= нещо на Български ''''''```````\"\"\"\"......!!!!!!?????? I've been 'told he's there, 'RE you sure? 'M not sure I'll make it, 'D you like some tea? We'Ve a'lL"

        chktok = tokenizer.encode(chktxt)
        chkhsh = sha256(str(chktok).encode()).hexdigest()

        logger.debug(f"chktok: {chktok}")
        logger.debug(f"chkhsh: {chkhsh}")

        res = None

        # NOTE: if you get an error here, you need to update the convert_hf_to_gguf_update.py script
        #       or pull the latest version of the model from Huggingface
        #       don't edit the hashes manually!
        if chkhsh == "0ef9807a4087ebef797fc749390439009c3b9eda9ad1a097abbe738f486c01e5":
            # ref: https://huggingface.co/meta-llama/Meta-Llama-3-8B
            res = "llama-bpe"
        if chkhsh == "049ecf7629871e3041641907f3de7c733e4dbfdc736f57d882ba0b0845599754":
            # ref: https://huggingface.co/deepseek-ai/deepseek-llm-7b-base
            res = "deepseek-llm"
        if chkhsh == "347715f544604f9118bb75ed199f68779f423cabb20db6de6f31b908d04d7821":
            # ref: https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-base
            res = "deepseek-coder"
        if chkhsh == "8aeee3860c56296a157a1fe2fad249ec40aa59b1bb5709f4ade11c4e6fe652ed":
            # ref: https://huggingface.co/tiiuae/falcon-7b
            res = "falcon"
        if chkhsh == "0876d13b50744004aa9aeae05e7b0647eac9d801b5ba4668afc01e709c15e19f":
            # ref: https://huggingface.co/BAAI/bge-small-en-v1.5
            res = "bert-bge"
        if chkhsh == "8e62295832751ca1e8f92f2226f403dea30dc5165e448b5bfa05af5340c64ec7":
            # ref: https://huggingface.co/BAAI/bge-large-zh-v1.5
            res = "bert-bge-large"
        if chkhsh == "b6dc8df998e1cfbdc4eac8243701a65afe638679230920b50d6f17d81c098166":
            # ref: https://huggingface.co/mosaicml/mpt-7b
            res = "mpt"
        if chkhsh == "35d91631860c815f952d711435f48d356ebac988362536bed955d43bfa436e34":
            # ref: https://huggingface.co/bigcode/starcoder2-3b
            res = "starcoder"
        if chkhsh == "3ce83efda5659b07b1ad37ca97ca5797ea4285d9b9ab0dc679e4a720c9da7454":
            # ref: https://huggingface.co/openai-community/gpt2
            res = "gpt-2"
        if chkhsh == "32d85c31273f8019248f2559fed492d929ea28b17e51d81d3bb36fff23ca72b3":
            # ref: https://huggingface.co/stabilityai/stablelm-2-zephyr-1_6b
            res = "stablelm2"
        if chkhsh == "6221ad2852e85ce96f791f476e0b390cf9b474c9e3d1362f53a24a06dc8220ff":
            # ref: https://huggingface.co/smallcloudai/Refact-1_6-base
            res = "refact"
        if chkhsh == "9c2227e4dd922002fb81bde4fc02b0483ca4f12911410dee2255e4987644e3f8":
            # ref: https://huggingface.co/CohereForAI/c4ai-command-r-v01
            res = "command-r"
        if chkhsh == "e636dc30a262dcc0d8c323492e32ae2b70728f4df7dfe9737d9f920a282b8aea":
            # ref: https://huggingface.co/Qwen/Qwen1.5-7B
            res = "qwen2"
        if chkhsh == "b6dc8df998e1cfbdc4eac8243701a65afe638679230920b50d6f17d81c098166":
            # ref: https://huggingface.co/allenai/OLMo-1.7-7B-hf
            res = "olmo"
        if chkhsh == "a8594e3edff7c29c003940395316294b2c623e09894deebbc65f33f1515df79e":
            # ref: https://huggingface.co/databricks/dbrx-base
            res = "dbrx"
        if chkhsh == "c7699093ba4255a91e702aa38a596aa81669f3525dae06c2953267dde580f448":
            # ref: https://huggingface.co/jinaai/jina-reranker-v1-tiny-en
            res = "jina-v1-en"
        if chkhsh == "0876d13b50744004aa9aeae05e7b0647eac9d801b5ba4668afc01e709c15e19f":
            # ref: https://huggingface.co/jinaai/jina-embeddings-v2-base-en
            res = "jina-v2-en"
        if chkhsh == "171aeeedd6fb548d418a7461d053f11b6f1f1fc9b387bd66640d28a4b9f5c643":
            # ref: https://huggingface.co/jinaai/jina-embeddings-v2-base-es
            res = "jina-v2-es"
        if chkhsh == "27949a2493fc4a9f53f5b9b029c82689cfbe5d3a1929bb25e043089e28466de6":
            # ref: https://huggingface.co/jinaai/jina-embeddings-v2-base-de
            res = "jina-v2-de"
        if chkhsh == "c136ed14d01c2745d4f60a9596ae66800e2b61fa45643e72436041855ad4089d":
            # ref: https://huggingface.co/abacusai/Smaug-Llama-3-70B-Instruct
            res = "smaug-bpe"
        if chkhsh == "c7ea5862a53e4272c035c8238367063e2b270d51faa48c0f09e9d5b54746c360":
            # ref: https://huggingface.co/LumiOpen/Poro-34B-chat
            res = "poro-chat"
        if chkhsh == "7967bfa498ade6b757b064f31e964dddbb80f8f9a4d68d4ba7998fcf281c531a":
            # ref: https://huggingface.co/jinaai/jina-embeddings-v2-base-code
            res = "jina-v2-code"
        if chkhsh == "b6e8e1518dc4305be2fe39c313ed643381c4da5db34a98f6a04c093f8afbe99b":
            # ref: https://huggingface.co/THUDM/glm-4-9b-chat
            res = "chatglm-bpe"
        if chkhsh == "7fc505bd3104ca1083b150b17d088b59534ede9bde81f0dd2090967d7fe52cee":
            # ref: https://huggingface.co/LumiOpen/Viking-7B
            res = "viking"
        if chkhsh == "b53802fb28e26d645c3a310b34bfe07da813026ec7c7716883404d5e0f8b1901":
            # ref: https://huggingface.co/core42/jais-13b
            res = "jais"
        if chkhsh == "7b3e7548e4308f52a76e8229e4e6cc831195d0d1df43aed21ac6c93da05fec5f":
            # ref: https://huggingface.co/WisdomShell/CodeShell-7B
            res = "codeshell"
        if chkhsh == "63b97e4253352e6f357cc59ea5b583e3a680eaeaf2632188c2b952de2588485e":
            # ref: https://huggingface.co/mistralai/Mistral-Nemo-Base-2407
            res = "tekken"
        if chkhsh == "855059429035d75a914d1eda9f10a876752e281a054a7a3d421ef0533e5b6249":
            # ref: https://huggingface.co/HuggingFaceTB/SmolLM-135M
            res = "smollm"
        if chkhsh == "3c30d3ad1d6b64202cd222813e7736c2db6e1bd6d67197090fc1211fbc612ae7":
            # ref: https://huggingface.co/bigscience/bloom
            res = "bloom"
        if chkhsh == "bc01ce58980e1db43859146dc51b1758b3b88729b217a74792e9f8d43e479d21":
            # ref: https://huggingface.co/TurkuNLP/gpt3-finnish-small
            res = "gpt3-finnish"
        if chkhsh == "4e2b24cc4770243d65a2c9ec19770a72f08cffc161adbb73fcbb6b7dd45a0aae":
            # ref: https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct
            res = "exaone"
        if chkhsh == "fcace8b9cac38ce847670c970cd5892031a753a1ef381abd1d9af00f713da085":
            # ref: https://huggingface.co/microsoft/phi-2
            res = "phi-2"
        if chkhsh == "60824e3c0d9401f89943cbb2fff727f0e2d4c545ba4df2d6e4f09a6db0f5b450":
            # ref: https://huggingface.co/facebook/chameleon-7b
            res = "chameleon"
        if chkhsh == "1431a23e583c97432bc230bff598d103ddb5a1f89960c8f1d1051aaa944d0b35":
            # ref: https://huggingface.co/sapienzanlp/Minerva-7B-base-v1.0
            res = "minerva-7b"
        if chkhsh == "8b5a93ed704057481f240da0be7e7dca721d7f8f4755263b6807227a2cbeae65":
            # ref: https://huggingface.co/sentence-transformers/stsb-roberta-base
            res = "roberta-bpe"

        if res is None:
            logger.warning("\n")
            logger.warning(
                "**************************************************************************************"
            )
            logger.warning("** WARNING: The BPE pre-tokenizer was not recognized!")
            logger.warning("**          There are 2 possible reasons for this:")
            logger.warning(
                "**          - the model has not been added to convert_hf_to_gguf_update.py yet"
            )
            logger.warning(
                "**          - the pre-tokenization config has changed upstream"
            )
            logger.warning(
                "**          Check your model files and convert_hf_to_gguf_update.py and update them accordingly."
            )
            logger.warning(
                "** ref:     https://github.com/ggerganov/llama.cpp/pull/6920"
            )
            logger.warning("**")
            logger.warning(f"** chkhsh:  {chkhsh}")
            logger.warning(
                "**************************************************************************************"
            )
            logger.warning("\n")
            raise NotImplementedError(
                "BPE pre-tokenizer was not recognized - update get_vocab_base_pre()"
            )

        logger.debug(f"tokenizer.ggml.pre: {repr(res)}")
        logger.debug(f"chkhsh: {chkhsh}")

        return res
        # Marker: End get_vocab_base_pre

    def _set_vocab_gpt2(self) -> None:
        tokens, toktypes, tokpre = self.get_vocab_base()
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        special_vocab.add_to_gguf(self.gguf_writer)

    def _set_vocab_sentencepiece(self, add_to_gguf=True):
        tokens, scores, toktypes = self._create_vocab_sentencepiece()

        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)

    def _create_vocab_sentencepiece(self):
        # Third Party
        from sentencepiece import SentencePieceProcessor

        tokenizer_path = self.dir_model / "tokenizer.model"

        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"File not found: {tokenizer_path}")

        tokenizer = SentencePieceProcessor()
        tokenizer.LoadFromFile(str(tokenizer_path))

        vocab_size = self.hparams.get("vocab_size", tokenizer.vocab_size())

        tokens: list[bytes] = [f"[PAD{i}]".encode("utf-8") for i in range(vocab_size)]
        scores: list[float] = [-10000.0] * vocab_size
        toktypes: list[int] = [SentencePieceTokenTypes.UNUSED] * vocab_size

        for token_id in range(tokenizer.vocab_size()):
            piece = tokenizer.IdToPiece(token_id)
            text = piece.encode("utf-8")
            score = tokenizer.GetScore(token_id)

            toktype = SentencePieceTokenTypes.NORMAL
            if tokenizer.IsUnknown(token_id):
                toktype = SentencePieceTokenTypes.UNKNOWN
            elif tokenizer.IsControl(token_id):
                toktype = SentencePieceTokenTypes.CONTROL
            elif tokenizer.IsUnused(token_id):
                toktype = SentencePieceTokenTypes.UNUSED
            elif tokenizer.IsByte(token_id):
                toktype = SentencePieceTokenTypes.BYTE

            tokens[token_id] = text
            scores[token_id] = score
            toktypes[token_id] = toktype

        added_tokens_file = self.dir_model / "added_tokens.json"
        if added_tokens_file.is_file():
            with open(added_tokens_file, "r", encoding="utf-8") as f:
                added_tokens_json = json.load(f)
                for key in added_tokens_json:
                    token_id = added_tokens_json[key]
                    if token_id >= vocab_size:
                        logger.warning(
                            f"ignore token {token_id}: id is out of range, max={vocab_size - 1}"
                        )
                        continue

                    tokens[token_id] = key.encode("utf-8")
                    scores[token_id] = -1000.0
                    toktypes[token_id] = SentencePieceTokenTypes.USER_DEFINED

        tokenizer_config_file = self.dir_model / "tokenizer_config.json"
        if tokenizer_config_file.is_file():
            with open(tokenizer_config_file, "r", encoding="utf-8") as f:
                tokenizer_config_json = json.load(f)
                added_tokens_decoder = tokenizer_config_json.get(
                    "added_tokens_decoder", {}
                )
                for token_id, token_data in added_tokens_decoder.items():
                    token_id = int(token_id)
                    token: str = token_data["content"]
                    if toktypes[token_id] != SentencePieceTokenTypes.UNUSED:
                        if tokens[token_id] != token.encode("utf-8"):
                            logger.warning(
                                f'replacing token {token_id}: {tokens[token_id].decode("utf-8")!r} -> {token!r}'
                            )
                    if token_data.get("special") or self.does_token_look_special(token):
                        toktypes[token_id] = SentencePieceTokenTypes.CONTROL
                    else:
                        token = token.replace(
                            b"\xe2\x96\x81".decode("utf-8"), " "
                        )  # pre-normalize user-defined spaces
                        toktypes[token_id] = SentencePieceTokenTypes.USER_DEFINED

                    scores[token_id] = -1000.0
                    tokens[token_id] = token.encode("utf-8")

        if vocab_size > len(tokens):
            pad_count = vocab_size - len(tokens)
            logger.debug(
                f"Padding vocab with {pad_count} token(s) - [PAD1] through [PAD{pad_count}]"
            )
            for i in range(1, pad_count + 1):
                tokens.append(bytes(f"[PAD{i}]", encoding="utf-8"))
                scores.append(-1000.0)
                toktypes.append(SentencePieceTokenTypes.UNUSED)

        return tokens, scores, toktypes

    def _set_vocab_llama_hf(self):
        vocab = gguf.LlamaHfVocab(self.dir_model)
        tokens = []
        scores = []
        toktypes = []

        for text, score, toktype in vocab.all_tokens():
            tokens.append(text)
            scores.append(score)
            toktypes.append(toktype)

        assert len(tokens) == vocab.vocab_size

        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)

    def _set_vocab_builtin(
        self, model_name: Literal["gpt-neox", "llama-spm"], vocab_size: int
    ):
        tokenizer_path = Path(sys.path[0]) / "models" / f"ggml-vocab-{model_name}.gguf"
        logger.warning(
            f"Using tokenizer from '{os.path.relpath(tokenizer_path, os.getcwd())}'"
        )
        vocab_reader = gguf.GGUFReader(tokenizer_path, "r")

        default_pre = "mpt" if model_name == "gpt-neox" else "default"

        field = vocab_reader.get_field(gguf.Keys.Tokenizer.MODEL)
        assert field  # tokenizer model
        self.gguf_writer.add_tokenizer_model(bytes(field.parts[-1]).decode("utf-8"))

        field = vocab_reader.get_field(gguf.Keys.Tokenizer.PRE)
        self.gguf_writer.add_tokenizer_pre(
            bytes(field.parts[-1]).decode("utf-8") if field else default_pre
        )

        field = vocab_reader.get_field(gguf.Keys.Tokenizer.LIST)
        assert field  # token list
        self.gguf_writer.add_token_list(
            [bytes(field.parts[i]) for i in field.data][:vocab_size]
        )

        if model_name == "llama-spm":
            field = vocab_reader.get_field(gguf.Keys.Tokenizer.SCORES)
            assert field  # token scores
            self.gguf_writer.add_token_scores(
                [field.parts[i].tolist()[0] for i in field.data][:vocab_size]
            )

        field = vocab_reader.get_field(gguf.Keys.Tokenizer.TOKEN_TYPE)
        assert field  # token types
        self.gguf_writer.add_token_types(
            [field.parts[i].tolist()[0] for i in field.data][:vocab_size]
        )

        if model_name != "llama-spm":
            field = vocab_reader.get_field(gguf.Keys.Tokenizer.MERGES)
            assert field  # token merges
            self.gguf_writer.add_token_merges(
                [bytes(field.parts[i]) for i in field.data]
            )

        if (field := vocab_reader.get_field(gguf.Keys.Tokenizer.BOS_ID)) is not None:
            self.gguf_writer.add_bos_token_id(field.parts[-1].tolist()[0])
        if (field := vocab_reader.get_field(gguf.Keys.Tokenizer.EOS_ID)) is not None:
            self.gguf_writer.add_eos_token_id(field.parts[-1].tolist()[0])
        if (field := vocab_reader.get_field(gguf.Keys.Tokenizer.UNK_ID)) is not None:
            self.gguf_writer.add_unk_token_id(field.parts[-1].tolist()[0])
        if (field := vocab_reader.get_field(gguf.Keys.Tokenizer.PAD_ID)) is not None:
            self.gguf_writer.add_pad_token_id(field.parts[-1].tolist()[0])
        if (field := vocab_reader.get_field(gguf.Keys.Tokenizer.ADD_BOS)) is not None:
            self.gguf_writer.add_add_bos_token(field.parts[-1].tolist()[0])
        if (field := vocab_reader.get_field(gguf.Keys.Tokenizer.ADD_EOS)) is not None:
            self.gguf_writer.add_add_eos_token(field.parts[-1].tolist()[0])


@Model.register(
    "LLaMAForCausalLM", "LlamaForCausalLM", "MistralForCausalLM", "MixtralForCausalLM"
)
class LlamaModel(Model):
    model_arch = gguf.MODEL_ARCH.LLAMA

    def set_vocab(self):
        try:
            self._set_vocab_sentencepiece()
        except FileNotFoundError:
            try:
                self._set_vocab_llama_hf()
            except (FileNotFoundError, TypeError):
                # Llama 3
                self._set_vocab_gpt2()

        # Apply to CodeLlama only (and ignore for Llama 3 with a vocab size of 128256)
        if self.hparams.get("vocab_size", 32000) == 32016:
            special_vocab = gguf.SpecialVocab(
                self.dir_model,
                load_merges=False,
                special_token_types=["prefix", "suffix", "middle", "eot"],
            )
            special_vocab._set_special_token("prefix", 32007)
            special_vocab._set_special_token("suffix", 32008)
            special_vocab._set_special_token("middle", 32009)
            special_vocab._set_special_token("eot", 32010)
            special_vocab.add_to_gguf(self.gguf_writer)

        tokenizer_config_file = self.dir_model / "tokenizer_config.json"
        if tokenizer_config_file.is_file():
            with open(tokenizer_config_file, "r", encoding="utf-8") as f:
                tokenizer_config_json = json.load(f)
                if "add_prefix_space" in tokenizer_config_json:
                    self.gguf_writer.add_add_space_prefix(
                        tokenizer_config_json["add_prefix_space"]
                    )

        # Apply to granite small models only
        if self.hparams.get("vocab_size", 32000) == 49152:
            self.gguf_writer.add_add_bos_token(False)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])

        if "head_dim" in hparams:
            rope_dim = hparams["head_dim"]
        else:
            rope_dim = hparams["hidden_size"] // hparams["num_attention_heads"]
        self.gguf_writer.add_rope_dimension_count(rope_dim)

        if (
            self.hparams.get("rope_scaling") is not None
            and "factor" in self.hparams["rope_scaling"]
        ):
            if self.hparams["rope_scaling"].get("type") == "linear":
                self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
                self.gguf_writer.add_rope_scaling_factor(
                    self.hparams["rope_scaling"]["factor"]
                )

    @staticmethod
    def permute(weights: Tensor, n_head: int, n_head_kv: int | None):
        if n_head_kv is not None and n_head != n_head_kv:
            n_head = n_head_kv
        return (
            weights.reshape(
                n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:]
            )
            .swapaxes(1, 2)
            .reshape(weights.shape)
        )

    _experts: list[dict[str, Tensor]] | None = None

    def modify_tensors(
        self, data_torch: Tensor, name: str, bid: int | None
    ) -> Iterable[tuple[str, Tensor]]:
        n_head = self.hparams["num_attention_heads"]
        n_kv_head = self.hparams.get("num_key_value_heads")

        if name.endswith(("q_proj.weight", "q_proj.bias")):
            data_torch = LlamaModel.permute(data_torch, n_head, n_head)
        if name.endswith(("k_proj.weight", "k_proj.bias")):
            data_torch = LlamaModel.permute(data_torch, n_head, n_kv_head)

        # process the experts separately
        if name.find("block_sparse_moe.experts") != -1:
            n_experts = self.hparams["num_local_experts"]

            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                tensors: list[tuple[str, Tensor]] = []

                # merge the experts into a single 3d tensor
                for wid in ["w1", "w2", "w3"]:
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.block_sparse_moe.experts.{xid}.{wid}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    data_torch = torch.stack(datas, dim=0)

                    merged_name = f"layers.{bid}.feed_forward.experts.{wid}.weight"

                    new_name = self.map_tensor_name(merged_name)

                    tensors.append((new_name, data_torch))
                return tensors
            else:
                return []

        return [(self.map_tensor_name(name), data_torch)]

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        if rope_scaling := self.find_hparam(["rope_scaling"], optional=True):
            if rope_scaling.get("rope_type", "").lower() == "llama3":
                base = self.hparams.get("rope_theta", 10000.0)
                dim = self.hparams.get(
                    "head_dim",
                    self.hparams["hidden_size"] // self.hparams["num_attention_heads"],
                )
                freqs = 1.0 / (
                    base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
                )

                factor = rope_scaling.get("factor", 8.0)
                low_freq_factor = rope_scaling.get("low_freq_factor", 1.0)
                high_freq_factor = rope_scaling.get("high_freq_factor", 4.0)
                old_context_len = self.hparams.get(
                    "original_max_position_embeddings", 8192
                )

                low_freq_wavelen = old_context_len / low_freq_factor
                high_freq_wavelen = old_context_len / high_freq_factor
                assert low_freq_wavelen != high_freq_wavelen

                rope_factors = []
                for freq in freqs:
                    wavelen = 2 * math.pi / freq
                    if wavelen < high_freq_wavelen:
                        rope_factors.append(1)
                    elif wavelen > low_freq_wavelen:
                        rope_factors.append(factor)
                    else:
                        smooth = (old_context_len / wavelen - low_freq_factor) / (
                            high_freq_factor - low_freq_factor
                        )
                        rope_factors.append(1 / ((1 - smooth) / factor + smooth))

                yield (
                    self.format_tensor_name(gguf.MODEL_TENSOR.ROPE_FREQS),
                    torch.tensor(rope_factors, dtype=torch.float32),
                )

    def prepare_tensors(self):
        super().prepare_tensors()

        if self._experts is not None:
            # flatten `list[dict[str, Tensor]]` into `list[str]`
            experts = [k for d in self._experts for k in d]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")


###### CONVERSION LOGIC ######


# tree of lazy tensors
class LazyTorchTensor(gguf.LazyBase):
    _tensor_type = torch.Tensor
    # to keep the type-checker happy
    dtype: torch.dtype
    shape: torch.Size

    # only used when converting a torch.Tensor to a np.ndarray
    _dtype_map: dict[torch.dtype, type] = {
        torch.float16: np.float16,
        torch.float32: np.float32,
    }

    # used for safetensors slices
    # ref: https://github.com/huggingface/safetensors/blob/079781fd0dc455ba0fe851e2b4507c33d0c0d407/bindings/python/src/lib.rs#L1046
    # TODO: uncomment U64, U32, and U16, ref: https://github.com/pytorch/pytorch/issues/58734
    _dtype_str_map: dict[str, torch.dtype] = {
        "F64": torch.float64,
        "F32": torch.float32,
        "BF16": torch.bfloat16,
        "F16": torch.float16,
        # "U64": torch.uint64,
        "I64": torch.int64,
        # "U32": torch.uint32,
        "I32": torch.int32,
        # "U16": torch.uint16,
        "I16": torch.int16,
        "U8": torch.uint8,
        "I8": torch.int8,
        "BOOL": torch.bool,
        "F8_E4M3": torch.float8_e4m3fn,
        "F8_E5M2": torch.float8_e5m2,
    }

    def numpy(self) -> gguf.LazyNumpyTensor:
        dtype = self._dtype_map[self.dtype]
        return gguf.LazyNumpyTensor(
            meta=gguf.LazyNumpyTensor.meta_with_dtype_and_shape(dtype, self.shape),
            args=(self,),
            func=(lambda s: s.numpy()),
        )

    @classmethod
    def meta_with_dtype_and_shape(
        cls, dtype: torch.dtype, shape: tuple[int, ...]
    ) -> Tensor:
        return torch.empty(size=shape, dtype=dtype, device="meta")

    @classmethod
    def from_safetensors_slice(cls, st_slice: Any) -> Tensor:
        dtype = cls._dtype_str_map[st_slice.get_dtype()]
        shape: tuple[int, ...] = tuple(st_slice.get_shape())
        lazy = cls(
            meta=cls.meta_with_dtype_and_shape(dtype, shape),
            args=(st_slice,),
            func=lambda s: s[:],
        )
        return cast(torch.Tensor, lazy)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        del types  # unused

        if kwargs is None:
            kwargs = {}

        if func is torch.Tensor.numpy:
            return args[0].numpy()

        return cls._wrap_fn(func)(*args, **kwargs)


def split_str_to_n_bytes(split_str: str) -> int:
    if split_str.endswith("K"):
        n = int(split_str[:-1]) * 1000
    elif split_str.endswith("M"):
        n = int(split_str[:-1]) * 1000 * 1000
    elif split_str.endswith("G"):
        n = int(split_str[:-1]) * 1000 * 1000 * 1000
    elif split_str.isnumeric():
        n = int(split_str)
    else:
        raise ValueError(
            f"Invalid split size: {split_str}, must be a number, optionally followed by K, M, or G"
        )

    if n < 0:
        raise ValueError(f"Invalid split size: {split_str}, must be positive")

    return n


def convert_model_to_gguf(
    model: Path,
    outfile: Path = None,
    metadata: Path = None,
    verbose: bool = False,
    split_max_tensors: int = 0,
    split_max_size: str = "0",
    use_temp_file: bool = False,
    # supported choices=["f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0", "auto"]
    outtype: str = "f16",
    bigendian: bool = False,
    no_lazy: bool = False,
    model_name: str = None,
    dry_run: bool = False,
    no_tensor_first_split: bool = False,
    vocab_only: bool = False,
) -> str:
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    dir_model = model

    if not dir_model.is_dir():
        logger.error(f"Error: {model} is not a directory")
        sys.exit(1)

    ftype_map: dict[str, gguf.LlamaFileType] = {
        "f32": gguf.LlamaFileType.ALL_F32,
        "f16": gguf.LlamaFileType.MOSTLY_F16,
        "bf16": gguf.LlamaFileType.MOSTLY_BF16,
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
        "tq1_0": gguf.LlamaFileType.MOSTLY_TQ1_0,
        "tq2_0": gguf.LlamaFileType.MOSTLY_TQ2_0,
        "auto": gguf.LlamaFileType.GUESSED,
    }

    is_split = split_max_tensors > 0 or split_max_size != "0"
    if use_temp_file and is_split:
        logger.error("Error: Cannot use temp file when splitting")
        sys.exit(1)

    if outfile is not None:
        fname_out = outfile
    else:
        fname_out = dir_model

    logger.info(f"Loading model: {dir_model.name}")

    hparams = Model.load_hparams(dir_model)

    with torch.inference_mode():
        output_type = ftype_map[outtype]
        model_architecture = hparams["architectures"][0]

        try:
            model_class = Model.from_model_architecture(model_architecture)
        except NotImplementedError:
            logger.error(f"Model {model_architecture} is not supported")
            sys.exit(1)

        model_instance = model_class(
            dir_model=dir_model,
            ftype=output_type,
            fname_out=fname_out,
            is_big_endian=bigendian,
            use_temp_file=use_temp_file,
            eager=no_lazy,
            metadata_override=metadata,
            model_name=model_name,
            split_max_tensors=split_max_tensors,
            split_max_size=split_str_to_n_bytes(split_max_size),
            dry_run=dry_run,
            small_first_shard=no_tensor_first_split,
        )

        out_path = ""
        if vocab_only:
            logger.info("Exporting model vocab...")
            model_instance.write_vocab()
            outpath = model_instance.fname_out
            logger.info(f"Model vocab successfully exported to {outpath}")
        else:
            logger.info("Exporting model...")
            model_instance.write()
            out_path = (
                f"{model_instance.fname_out.parent}{os.sep}"
                if is_split
                else model_instance.fname_out
            )
            logger.info(f"Model successfully exported to {out_path}")

    return out_path
