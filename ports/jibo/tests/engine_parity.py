"""Compare jibo-decider's answers with the Python engine's on evaluation/device_parity.py's requests.

    python ports/jibo/tests/engine_parity.py JIBO_DECIDER MODEL.jdw TOKENIZER.jdt CHECKPOINT
        [--window N] [--no-prefix-cache] [--out FILE]

The reference is the Python engine on the CPU in fp32 (the serving path, adapter unmerged).
Both see the same 18 requests: three tickets at three state lengths (about 40 to 3,000 tokens),
each asked one question (the whole-prompt path) and five (the shared-prefix path), 54 answers.
Reports the largest probability difference, how many chosen answers changed, whether token
counts match, and whether the rounded response JSON is identical.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "evaluation"))

from device_parity import chosen, probabilities, requests  # noqa: E402

from strands_decider.infer import EngineConfig, SystemOneEngine  # noqa: E402
from strands_decider.modeling import StrandsDeciderModel  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("jdw")
    ap.add_argument("jdt")
    ap.add_argument("checkpoint")
    ap.add_argument("--window", type=int, default=0, help="override max_length on both sides")
    ap.add_argument("--no-prefix-cache", action="store_true")
    ap.add_argument("--tol", type=float, default=1e-4, help="largest probability difference allowed")
    ap.add_argument("--out")
    args = ap.parse_args()

    model = StrandsDeciderModel.load(args.checkpoint)
    if args.window:
        model.config.max_length = args.window
    eng = SystemOneEngine(model, EngineConfig(device="cpu", use_prefix_cache=not args.no_prefix_cache))
    reqs = requests()
    worst, changed, total, token_mismatch, json_diff = 0.0, 0, 0, 0, 0
    report = {}
    for name, req in reqs.items():
        ref = json.loads(eng.evaluate(req).model_dump_json())
        cmd = [args.binary, "ask", args.jdw, args.jdt, "--raw"]
        if args.window:
            cmd += ["--window", str(args.window)]
        if args.no_prefix_cache:
            cmd.append("--no-prefix-cache")
        res = subprocess.run(cmd, input=req.model_dump_json().encode(), capture_output=True, check=True)
        lines = res.stdout.decode().strip().split("\n")
        got = json.loads(lines[0])
        raw = {d["name"]: dict(zip(d["labels"], d["probs"], strict=True)) for d in map(json.loads, lines[1:])}
        if got["usage"] != ref["usage"]:
            token_mismatch += 1
        for q, r in ref["answers"].items():
            p_ref = probabilities(r)
            p_raw = raw[q]
            d = max(abs(p_raw[k] - p_ref[k]) for k in p_ref)
            worst = max(worst, d)
            changed += chosen(got["answers"][q]) != chosen(r)
            total += 1
            if got["answers"][q] != r:
                json_diff += 1
        report[name] = {"ref": ref, "c": got, "latency_ms": got["latency_ms"]}
        print(f"{name:14s} tokens {ref['usage']['input_tokens']:5d}  C {got['latency_ms']:9.1f} ms")
    print(f"max |dp| {worst:.2e} (unrounded C against the engine's 4-decimal answers), "
          f"changed answers {changed}/{total}, token-count mismatches {token_mismatch}, "
          f"answers whose rounded JSON differs {json_diff}/{total}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)
    ok = worst <= args.tol + 5e-5 and changed == 0 and token_mismatch == 0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
