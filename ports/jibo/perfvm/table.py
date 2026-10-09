#!/usr/bin/env python3
"""table.py: the README's table of optimisation rounds, from perfvm/results/rounds.tsv (label, then
what changed) and each round's JSONL: G instructions per variant and workload, and the change
against the baseline."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
WL = ["prefill", "prefix_hit", "request3"]


def load(label):
    with open(os.path.join(HERE, "results", f"{label}.jsonl")) as fh:
        return {(r["variant"], r["workload"]): r["total"] for r in map(json.loads, fh)}


rounds = [line.rstrip("\n").split("\t", 1) for line in open(os.path.join(HERE, "results", "rounds.tsv")) if line.strip()]
base = load(rounds[0][0])
print("| round | change | " + " | ".join(f"{v} {w}" for v in ("plain", "NEON") for w in WL) + " |")
print("| --- | --- | " + " | ".join("---:" for _ in range(6)) + " |")
for label, what in rounds:
    r = load(label)
    cells = []
    for v in ("plain", "neon"):
        for w in WL:
            g = r[(v, w)] / 1e9
            d = (r[(v, w)] - base[(v, w)]) / base[(v, w)] * 100
            cells.append(f"{g:.2f}" + ("" if label == rounds[0][0] else f" ({d:+.0f}%)"))
    print(f"| `{label}` | {what} | " + " | ".join(cells) + " |")
