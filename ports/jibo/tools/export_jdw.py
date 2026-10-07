"""Export a decider checkpoint for the C runtime: model.jdw, tokenizer.jdt and a manifest.

    python ports/jibo/tools/export_jdw.py CHECKPOINT OUT_DIR [--weights f32|q8|q4] [--embed f32|bf16]

CHECKPOINT is anything StrandsDeciderModel.load accepts: a checkpoint directory or a Hub id.
The torso is loaded in fp32, the LoRA adapter is merged in fp32 (W + alpha/r * B A), and then
the layer matrices are written in the requested format. Norms, the small DeltaNet projections
and the pointer head stay fp32. The embedding table is only ever looked up, never multiplied;
bf16 stores it exactly, since it is not adapted and the base weights are bf16.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jdt import build_jdt
from jdw import tensors_from_model, write_jdw

from strands_decider.modeling import StrandsDeciderModel, checkpoint_dir


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_merged(checkpoint: str) -> StrandsDeciderModel:
    model = StrandsDeciderModel.load(checkpoint)
    model.eval()
    with torch.no_grad():
        model.torso.to(torch.float32)
        if model.config.use_lora:
            model.torso = model.torso.merge_and_unload()
    return model


def meta_for(model: StrandsDeciderModel, checkpoint: str, weights: str) -> dict[str, str]:
    c = model.torso.config
    cfg = model.config
    if cfg.head_type != "pointer":
        raise ValueError("the C runtime implements the pointer head only")
    by_kind = cfg.temperature_by_kind or {}
    rope = c.rope_parameters
    head_dim = c.head_dim
    meta = {
        "format": "strands-decider-jdw",
        "weights": weights,
        "checkpoint": checkpoint,
        "base_model": cfg.base_model,
        "base_revision": str(cfg.base_revision),
        "vocab_size": c.vocab_size,
        "hidden_size": c.hidden_size,
        "intermediate_size": c.intermediate_size,
        "num_layers": c.num_hidden_layers,
        "layer_types": ",".join("L" if t == "linear_attention" else "A" for t in c.layer_types),
        "num_heads": c.num_attention_heads,
        "num_kv_heads": c.num_key_value_heads,
        "head_dim": head_dim,
        "rotary_dim": int(head_dim * rope.get("partial_rotary_factor", 1.0)),
        "rope_theta": repr(float(rope["rope_theta"])),
        "rms_eps": repr(float(c.rms_norm_eps)),
        "lin_num_k_heads": c.linear_num_key_heads,
        "lin_num_v_heads": c.linear_num_value_heads,
        "lin_k_dim": c.linear_key_head_dim,
        "lin_v_dim": c.linear_value_head_dim,
        "conv_kernel": c.linear_conv_kernel_dim,
        "max_length": cfg.max_length,
        "pointer_dim": cfg.pointer_dim,
        "head_ln_eps": repr(float(model.head.norm.eps)),
        "temperature": repr(float(cfg.temperature)),
        "ordinal_smoothing": repr(float(cfg.ordinal_smoothing)),
    }
    for kind in ("noul", "choice", "score"):
        meta[f"temperature_{kind}"] = repr(float(by_kind.get(kind, cfg.temperature)))
    if c.hidden_act != "silu" or c.attention_bias or not getattr(c, "attn_output_gate", True):
        raise ValueError("unsupported torso variant")
    return {k: str(v) for k, v in meta.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("out_dir")
    ap.add_argument("--weights", choices=["f32", "q8", "q4"], default="f32")
    ap.add_argument("--embed", choices=["f32", "bf16"], default="bf16")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    src = checkpoint_dir(args.checkpoint)

    model = load_merged(args.checkpoint)
    jdt_path = os.path.join(args.out_dir, "tokenizer.jdt")
    with open(jdt_path, "wb") as fh:
        fh.write(build_jdt(model.tokenizer.backend_tokenizer.to_str()))
    meta = meta_for(model, args.checkpoint, args.weights)
    jdw_path = os.path.join(args.out_dir, "model.jdw")
    with torch.no_grad():
        jdw_sha = write_jdw(jdw_path, meta, tensors_from_model(
            model.torso, model.head, model.config, args.weights, args.embed))

    sources = {}
    for root, _, files in os.walk(src, followlinks=True):
        for f in sorted(files):
            p = os.path.join(root, f)
            sources[os.path.relpath(p, src)] = sha256_file(p)
    manifest = {
        "checkpoint": args.checkpoint,
        "weights": args.weights,
        "embed": args.embed,
        "model.jdw": {"sha256": jdw_sha, "bytes": os.path.getsize(jdw_path)},
        "tokenizer.jdt": {"sha256": sha256_file(jdt_path), "bytes": os.path.getsize(jdt_path)},
        "meta": meta,
        "source_files_sha256": sources,
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(json.dumps({k: manifest[k] for k in ("model.jdw", "tokenizer.jdt")}, indent=2))


if __name__ == "__main__":
    main()
