"""Quantised exports against the fp32 export, on held-out decider questions: KL, agreement, accuracy.

    python ports/jibo/tests/kl_eval.py JIBO_DECIDER REFERENCE_DIR CANDIDATE_DIR... [--per-file 30]
        [--window 1024] [--out FILE]

Each *_DIR is an export (model.jdw, tokenizer.jdt). Questions come from the evaluation files
under data/ (held out of training, and of GPTQ's calibration, which reads training files only),
--per-file rows spread evenly over each, asked one question per request. For every candidate it
reports the mean and largest KL divergence KL(reference || candidate) over the option
distributions, how often the chosen answer agrees with the reference's, the largest probability
difference, and against the gold labels the accuracy and Brier score of both. KL on held-out data
is the yardstick Unsloth's dynamic quants are judged by; here the distributions are the
decider's answers rather than next-token probabilities.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

from strands_decider.data.format import read_jsonl  # noqa: E402
from strands_decider.schema import SystemOneRequest  # noqa: E402


def gold_label(ex) -> str:
    if ex.kind == "noul":
        return ("false", "true")[ex.label]
    if ex.kind == "choice":
        return ex.options[ex.label][0]
    return str(ex.label)


def ask(binary: str, export: str, body: str, window: int) -> dict[str, float]:
    res = subprocess.run([binary, "ask", os.path.join(export, "model.jdw"), os.path.join(export, "tokenizer.jdt"),
                          "--raw", "--window", str(window)], input=body.encode(), capture_output=True, check=True)
    raw = json.loads(res.stdout.decode().strip().split("\n")[1])
    return dict(zip(raw["labels"], raw["probs"], strict=True))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("reference")
    ap.add_argument("candidates", nargs="+")
    ap.add_argument("--per-file", type=int, default=30)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--jobs", type=int, default=1, help="requests run side by side (each single-threaded)")
    ap.add_argument("--out")
    args = ap.parse_args()

    items = []
    files = [*sorted(glob.glob(os.path.join(REPO, "data", "*eval*.jsonl"))),
             os.path.join(REPO, "data", "holdout_v5_norule.jsonl")]
    for path in files:
        if "para_pairs" in path or "flips" in path:
            continue  # pairs and flips are other kinds of evaluation
        rows = list(read_jsonl(path))
        for ex in rows[:: max(1, len(rows) // args.per_file)][: args.per_file]:
            body = SystemOneRequest(state=ex.state, questions={"q": ex.to_question()}).model_dump_json()
            items.append((os.path.basename(path), body, gold_label(ex)))

    def run(export: str) -> list[dict[str, float]]:
        with ThreadPoolExecutor(args.jobs) as pool:
            return list(pool.map(lambda it: ask(args.binary, export, it[1], args.window), items))

    ref = run(args.reference)
    report = {"n": len(items), "window": args.window, "reference": args.reference, "candidates": {}}

    def acc_brier(dists: list[dict[str, float]]) -> tuple[float, float]:
        acc = sum(max(d, key=d.get) == g for d, (_, _, g) in zip(dists, items, strict=True)) / len(items)
        brier = sum(sum((p - (k == g)) ** 2 for k, p in d.items())
                    for d, (_, _, g) in zip(dists, items, strict=True)) / len(items)
        return acc, brier

    ra, rb = acc_brier(ref)
    print(f"{len(items)} held-out questions, window {args.window}; reference accuracy {ra:.3f}, Brier {rb:.4f}")
    for cand in args.candidates:
        got = run(cand)
        kls, dps, agree = [], [], 0
        for r, g in zip(ref, got, strict=True):
            kls.append(sum(p * math.log(max(p, 1e-12) / max(g[k], 1e-12)) for k, p in r.items()))
            dps.append(max(abs(p - g[k]) for k, p in r.items()))
            agree += max(r, key=r.get) == max(g, key=g.get)
        a, b = acc_brier(got)
        row = {"kl_mean": sum(kls) / len(kls), "kl_max": max(kls), "agreement": agree / len(items),
               "max_abs_dp": max(dps), "accuracy": a, "brier": b,
               "accuracy_delta": a - ra, "brier_delta": b - rb}
        report["candidates"][cand] = row
        print(f"{cand}: KL mean {row['kl_mean']:.5f} max {row['kl_max']:.4f}, agreement {row['agreement']:.3f}, "
              f"max |dp| {row['max_abs_dp']:.3f}, accuracy {a:.3f} ({a - ra:+.3f}), Brier {b:.4f} ({b - rb:+.4f})")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
