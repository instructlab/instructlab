# Copyright © 2023 Apple Inc.

import mlx.core as mx
import utils
import numpy as np
import torch
import argparse
from pathlib import Path

def translate_weight_names(name):
    name = name.replace("blk.", "model.layers.")
    name = name.replace("ffn_gate", "mlp.gate_proj")
    name = name.replace("ffn_down", "mlp.down_proj")
    name = name.replace("ffn_up", "mlp.up_proj")
    name = name.replace("attn_q", "self_attn.q_proj")
    name = name.replace("attn_k", "self_attn.k_proj")
    name = name.replace("attn_v", "self_attn.v_proj")
    name = name.replace("attn_output", "self_attn.o_proj")
    name = name.replace("attn_norm", "input_layernorm")
    name = name.replace("ffn_norm", "post_attention_layernorm")
    name = name.replace("token_embd", "model.embed_tokens")
    name = name.replace("output_norm", "model.norm")
    name = name.replace("output", "lm_head")
    return name


def load(gguf_file: str, dest_path: str = None):
    # If the gguf_file exists, try to load model from it.
    if not Path(gguf_file).exists():
        raise ValueError(
            f"Could not find file {gguf_file}"
        )

    print(f"[INFO] Loading model from {gguf_file}")
    weights, metadata = mx.load(gguf_file, return_metadata=True)
    weights = {
        k: torch.from_numpy(np.array(v, copy=False, dtype="float16"))
        for k, v in weights.items()
    }
    weights = {translate_weight_names(k): v for k, v in weights.items()}
    utils.save_model(dest_path, weights)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument(
        "--gguf",
        type=str,
        help="The GGUF file to load (and optionally download).",
    )
    parser.add_argument(
        "--dest-path",
        type=str,
        default=None,
        help="The destination where the converted mlx model should be stored.",
    )
    args = parser.parse_args()
    load(args.gguf, args.dest_path)
