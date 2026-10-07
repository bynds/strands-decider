"""Export a decider checkpoint for the C runtime: model.jdw, tokenizer.jdt and a manifest.

    python ports/jibo/tools/export_jdw.py CHECKPOINT OUT_DIR [--weights f32|q8|q4] [--embed f32|bf16]
        [--gptq N [--calib-file FILE ...] [--q8-budget-mb M]] [--q8 down,o,...]

CHECKPOINT is anything StrandsDeciderModel.load accepts: a checkpoint directory or a Hub id.
The torso is loaded in fp32, the LoRA adapter is merged in fp32 (W + alpha/r * B A), and then
the layer matrices are written in the requested format. Norms, the small DeltaNet projections
and the pointer head stay fp32. The embedding table is only ever looked up, never multiplied;
bf16 stores it exactly, since it is not adapted and the base weights are bf16.

With --gptq N, the q4 matrices are quantised with GPTQ (tools/gptq.py), calibrated on N
prompts rendered from the calibration files (by default the training corpus's
data/train_v5.jsonl, data/multistep_v14.jsonl and data/generated_v16p.jsonl, read in turn),
each cut to its first 512 tokens. The format, and so the runtime, is the same. --q8 names
matrix types (qkv, z, out, q, k, v, o, gate, up, down) to keep at 8 bits in a q4 export.

--q8-budget-mb M allocates bit widths per matrix, in the manner of Unsloth's dynamic quants: every
matrix is quantised to q4 with GPTQ and scored by its Hessian-weighted output error (gptq.
hessian_error), and the matrices that remove the most error per byte are raised to q8 until M
extra megabytes are spent. The choice and the scores go into the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from gptq import (  # noqa: E402
    allocate_q8,
    calibration_hessians,
    hessian_error,
    pack_q4,
    quantise_gptq,
)
from jdt import build_jdt  # noqa: E402
from jdw import (  # noqa: E402
    Q4,
    SHARED_INPUT,
    dequantise,
    layer_matrices,
    tensors_from_model,
    write_jdw,
)

from strands_decider.data.format import read_jsonl  # noqa: E402
from strands_decider.modeling import StrandsDeciderModel, checkpoint_dir  # noqa: E402
from strands_decider.prompting import build_prompt  # noqa: E402

DEFAULT_CALIB = ["data/train_v5.jsonl", "data/multistep_v14.jsonl", "data/generated_v16p.jsonl"]


def calibration_prompts(model: StrandsDeciderModel, files: list[str], n: int, max_tokens: int = 512):
    """n token sequences: rows spread evenly over each file in turn, rendered as the engine renders."""
    per = max(1, n // len(files))
    out: list[list[int]] = []
    for path in files:
        rows = list(read_jsonl(path))
        step = max(1, len(rows) // per)
        for ex in rows[::step][:per]:
            prompt, _ = build_prompt(ex.state, ex.to_question())
            out.append(model.tokenizer(prompt, add_special_tokens=True)["input_ids"][:max_tokens])
    return out[:n]


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


def dynamic_allocation(torso: torch.nn.Module, hessians: dict[str, torch.Tensor], budget_mb: float):
    """GPTQ every layer matrix to q4, score each by its Hessian-weighted error, and choose the
    matrices to raise to q8 within budget_mb extra megabytes. Returns the packed q4 bytes (reused
    for the matrices left at q4) and the allocation for the manifest."""
    packed, scores, extra = {}, {}, {}
    for name, w in layer_matrices(torso):
        a = w.detach().float().numpy()
        layer, short = name.split(".", 1)
        h = hessians[f"{layer}.{SHARED_INPUT[short]}"].numpy()
        packed[name] = pack_q4(*quantise_gptq(a, h))
        scores[name] = hessian_error(a, dequantise(packed[name], Q4, *a.shape), h)
        extra[name] = a.size // 32 * (34 - 18)
    chosen = allocate_q8(scores, extra, int(budget_mb * 1e6))
    return packed, {"budget_mb": budget_mb, "q8": chosen,
                    "extra_mb": sum(extra[n] for n in chosen) / 1e6, "q4_scores": scores}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("out_dir")
    ap.add_argument("--weights", choices=["f32", "q8", "q4"], default="f32")
    ap.add_argument("--embed", choices=["f32", "bf16"], default="bf16")
    ap.add_argument("--gptq", type=int, default=0, metavar="N", help="GPTQ q4 with N calibration prompts")
    ap.add_argument("--calib-file", action="append", default=None)
    ap.add_argument("--q8", default="", help="comma-separated matrix types kept at 8 bits")
    ap.add_argument("--q8-budget-mb", type=float, default=0.0,
                    help="with --gptq: raise the most sensitive matrices to q8 within this many MB")
    args = ap.parse_args()
    q8 = frozenset(x for x in args.q8.split(",") if x)
    os.makedirs(args.out_dir, exist_ok=True)
    src = checkpoint_dir(args.checkpoint)

    model = load_merged(args.checkpoint)
    jdt_path = os.path.join(args.out_dir, "tokenizer.jdt")
    with open(jdt_path, "wb") as fh:
        fh.write(build_jdt(model.tokenizer.backend_tokenizer.to_str()))
    meta = meta_for(model, args.checkpoint, args.weights)
    if q8:
        meta["q8_matrices"] = ",".join(sorted(q8))
    hessians = None
    if args.gptq and args.weights == "q4":
        files = args.calib_file or [os.path.join(REPO, f) for f in DEFAULT_CALIB]
        prompts = calibration_prompts(model, files, args.gptq)
        hessians = calibration_hessians(model.torso, prompts)
        meta["gptq"] = f"{len(prompts)} prompts, {sum(map(len, prompts))} tokens, from " + ",".join(
            os.path.relpath(f, REPO) for f in files)
    packed: dict[str, bytes] = {}
    allocation = None
    if hessians is not None and args.q8_budget_mb > 0:
        packed, allocation = dynamic_allocation(model.torso, hessians, args.q8_budget_mb)
        chosen = allocation["q8"]
        q8 = q8 | frozenset(chosen)
        meta["q8_matrices"] = ",".join(chosen)
        print(f"[export] {len(chosen)} of {len(packed)} matrices raised to q8, "
              f"{allocation['extra_mb']:.1f} MB: {', '.join(chosen)}")
    jdw_path = os.path.join(args.out_dir, "model.jdw")
    with torch.no_grad():
        jdw_sha = write_jdw(jdw_path, meta, tensors_from_model(
            model.torso, model.head, model.config, args.weights, args.embed, hessians, q8, packed))

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
        "allocation": allocation,
        "source_files_sha256": sources,
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(json.dumps({k: manifest[k] for k in ("model.jdw", "tokenizer.jdt")}, indent=2))


if __name__ == "__main__":
    main()
