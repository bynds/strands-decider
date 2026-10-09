#!/usr/bin/env python3
"""python_check.py: compare traces of the C runtime (tests/trace.c) with the original Python engine.

    python perfvm/python_check.py CHECKPOINT TRACE... [--layers N] [--json OUT]
        [--hidden-tol T --prob-tol P]

The reference is the serving path, strands_decider's SystemOneEngine on CPU in fp32 with the
LoRA adapter unmerged, run on what the traces hold: every layer's output over the trace's own
token ids, the final normed hidden states, the eight one-token continuations, and the engine's
answers to request1 and to request3 with and without the prefix cache (unrounded: the
probabilities the engine rounds to four decimals). --layers N cuts the torso to its first N
layers, to match a model written by tools/truncate_jdw.py.

Per trace it prints, for the hidden states, the largest absolute difference relative to the
largest activation (per layer, the worst one shown) and, for the answers, the largest absolute
difference in probability, the chosen option per question and the token count. A trace of an
f32 export differs from Python only by the C runtime's arithmetic; one of a q4 export also by
quantisation. With --hidden-tol and --prob-tol (for f32 traces) it fails when a hidden state or a
probability differs by more, or a choice or a token count differs at all.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

from strands_decider import infer  # noqa: E402
from strands_decider.infer import EngineConfig, SystemOneEngine  # noqa: E402
from strands_decider.modeling import StrandsDeciderModel  # noqa: E402
from strands_decider.schema import SystemOneRequest  # noqa: E402


def sections(path: str) -> dict[str, bytes]:
    data = open(path, "rb").read()
    out, pos = {}, 0
    while pos < len(data):
        name = data[pos:pos + 16].rstrip(b"\0").decode()
        (n,) = struct.unpack_from("<I", data, pos + 16)
        out[name] = data[pos + 20:pos + 20 + n]
        pos += 20 + n
    return out


def torso_layers(torso: torch.nn.Module) -> torch.nn.ModuleList:
    return torso.base_model.model.layers if hasattr(torso, "base_model") else torso.layers


def load(checkpoint: str, layers: int) -> StrandsDeciderModel:
    model = StrandsDeciderModel.load(checkpoint)
    model.eval()
    model.torso.to(torch.float32)
    if layers:
        inner = model.torso.base_model.model if hasattr(model.torso, "base_model") else model.torso
        inner.layers = torch.nn.ModuleList(list(inner.layers)[:layers])
        inner.config.num_hidden_layers = layers
        inner.config.layer_types = list(inner.config.layer_types)[:layers]
    return model


def reference(model: StrandsDeciderModel, ids: list[int]) -> dict[str, np.ndarray]:
    """Every layer's output and the final hidden states over ids plus the trace's eight steps."""
    n = len(ids)
    steps = [ids[(j * 37) % n] for j in range(8)]
    full = ids + steps
    outs: list[torch.Tensor] = []
    hooks = [lyr.register_forward_hook(lambda _m, _i, o: outs.append(o[0] if isinstance(o, tuple) else o))
             for lyr in torso_layers(model.torso)]
    with torch.inference_mode():
        last = model.torso(input_ids=torch.tensor([full]), attention_mask=torch.ones(1, len(full), dtype=torch.long),
                           return_dict=True).last_hidden_state[0].float().numpy()
    for h in hooks:
        h.remove()
    ref = {f"layer{i}": o[0, :n].float().numpy() for i, o in enumerate(outs)}
    ref["final"] = last[:n]
    for j in range(8):
        ref[f"step{j}"] = last[n + j]
    return ref


def answers(model: StrandsDeciderModel, path: str, prefix_cache: bool) -> tuple[list[list[float]], int]:
    """The engine's unrounded probabilities per question, in rendered order, and its token count."""
    rows: list[list[float]] = []
    # the engine rounds to four decimals in _to_answer; wrapping it reads the probabilities it gets
    real = infer._to_answer  # noqa: SLF001

    def capture(rq, probs, **kw):
        rows.append(list(probs))
        return real(rq, probs, **kw)

    infer._to_answer = capture  # noqa: SLF001
    try:
        req = SystemOneRequest.model_validate_json(open(path, encoding="utf-8").read())
        resp = SystemOneEngine(model, EngineConfig(device="cpu", use_prefix_cache=prefix_cache)).evaluate(req)
    finally:
        infer._to_answer = real  # noqa: SLF001
    return rows, resp.usage.input_tokens


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--json")
    ap.add_argument("--hidden-tol", type=float, default=0.0, help="fail above this (0: report only)")
    ap.add_argument("--prob-tol", type=float, default=0.0, help="fail above this (0: report only)")
    args = ap.parse_args()
    failed = False
    fx = os.path.join(HERE, "fixtures")
    model = load(args.checkpoint, args.layers)
    with torch.inference_mode():
        want = {"probs1": answers(model, os.path.join(fx, "request1.json"), True),
                "probs3": answers(model, os.path.join(fx, "request3.json"), True),
                "probs3_flat": answers(model, os.path.join(fx, "request3.json"), False)}
    cache: dict[tuple[int, ...], dict[str, np.ndarray]] = {}
    report = {}
    for path in args.traces:
        t = sections(path)
        ids = np.frombuffer(t["ids"], np.int32).tolist()
        key = tuple(ids)
        if key not in cache:
            cache[key] = reference(model, ids)
        ref = cache[key]
        hidden = ref["final"].shape[1]
        n = len(ids)
        got_layers: dict[int, list[np.ndarray]] = {}
        for name, b in t.items():
            if name.startswith("layer"):
                li, t0 = name[5:].split("@")
                got_layers.setdefault(int(li), []).append((int(t0), np.frombuffer(b, np.float32).reshape(-1, hidden)))
        rel = {}
        for li, parts in sorted(got_layers.items()):
            g = np.concatenate([p for _, p in sorted(parts, key=lambda x: x[0])])[:n]
            w = ref[f"layer{li}"]
            rel[f"layer{li}"] = float(np.abs(g - w).max() / np.abs(w).max())
        g = np.frombuffer(t["final"], np.float32).reshape(n, hidden)
        rel["final"] = float(np.abs(g - ref["final"]).max() / np.abs(ref["final"]).max())
        steps = np.stack([np.frombuffer(t[f"step{j}"], np.float32) for j in range(8)])
        ws = np.stack([ref[f"step{j}"] for j in range(8)])
        rel["steps"] = float(np.abs(steps - ws).max() / np.abs(ws).max())
        ans = {}
        for s, (rows, ntok) in want.items():
            got = np.frombuffer(t[s], np.float32)
            flat = np.array([p for r in rows for p in r], np.float64)
            dp = float(np.abs(got[:-1] - flat).max())
            same, i = True, 0
            for r in rows:
                same &= int(np.argmax(got[i:i + len(r)])) == int(np.argmax(r))
                i += len(r)
            ans[s] = {"max_abs_dp": dp, "same_choices": bool(same), "tokens": int(got[-1]), "tokens_ref": ntok}
        worst_layer = max((v, k) for k, v in rel.items() if k.startswith("layer"))
        report[path] = {"hidden_rel": rel, "answers": ans}
        if args.hidden_tol and max(rel.values()) > args.hidden_tol:
            failed = True
        for a in ans.values():
            if (args.prob_tol and a["max_abs_dp"] > args.prob_tol) or (
                    args.hidden_tol and (not a["same_choices"] or a["tokens"] != a["tokens_ref"])):
                failed = True
        print(f"{os.path.basename(path)}: hidden worst {worst_layer[1]} {worst_layer[0]:.2e}, final {rel['final']:.2e}, "
              f"steps {rel['steps']:.2e}; answers " + ", ".join(
                  f"{s} |dp| {a['max_abs_dp']:.2e}{'' if a['same_choices'] else ' CHOICE DIFFERS'}"
                  f"{'' if a['tokens'] == a['tokens_ref'] else ' TOKENS DIFFER'}" for s, a in ans.items()))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
