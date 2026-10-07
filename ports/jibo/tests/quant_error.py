"""How far do quantised exports move the torso's output? Hidden-state error against an fp32 export.

    python ports/jibo/tests/quant_error.py FWD_DUMP REFERENCE.jdw CHECKPOINT CANDIDATE.jdw... [--n 24]

Runs held-out prompts (rendered from the evaluation files under data/, which no calibration set
draws on) through tests/fwd_dump.c with each export, and reports, per candidate, the relative
error of the final hidden states, ||h - h_ref|| / ||h_ref||, over every token and at the last
one (the `<answer>` position the pointer head reads), averaged over prompts.
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

from strands_decider.data.format import read_jsonl  # noqa: E402
from strands_decider.modeling import checkpoint_dir  # noqa: E402
from strands_decider.prompting import build_prompt  # noqa: E402


def final_hidden(binary: str, jdw: str, ids: list[int], hidden: int, layers: int) -> np.ndarray:
    with tempfile.TemporaryDirectory() as d:
        np.asarray(ids, dtype=np.int32).tofile(os.path.join(d, "ids.bin"))
        subprocess.run([binary, jdw, os.path.join(d, "ids.bin"), os.path.join(d, "out.bin")], check=True)
        out = np.fromfile(os.path.join(d, "out.bin"), dtype=np.float32).reshape(layers + 1, len(ids), hidden)
    return out[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("reference")
    ap.add_argument("checkpoint")
    ap.add_argument("candidates", nargs="+")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(checkpoint_dir(args.checkpoint))
    meta = dict(line.split("=", 1) for line in open(args.reference, "rb").read(65536)
                .split(b"\n\0", 1)[0].decode("utf-8", "ignore").splitlines() if "=" in line)
    hidden, layers = int(meta["hidden_size"]), int(meta["num_layers"])
    files = sorted(glob.glob(os.path.join(REPO, "data", "*eval*.jsonl")))
    prompts: list[list[int]] = []
    per = max(1, args.n // max(1, len(files)))
    for path in files:
        rows = list(read_jsonl(path))
        for ex in rows[:: max(1, len(rows) // per)][:per]:
            p, _ = build_prompt(ex.state, ex.to_question())
            prompts.append(tok(p)["input_ids"][-args.max_tokens:])
    prompts = prompts[: args.n]
    refs = [final_hidden(args.binary, args.reference, ids, hidden, layers) for ids in prompts]
    print(f"{len(prompts)} held-out prompts, {sum(map(len, prompts))} tokens; reference {args.reference}")
    for cand in args.candidates:
        all_err, last_err = [], []
        for ids, ref in zip(prompts, refs, strict=True):
            h = final_hidden(args.binary, cand, ids, hidden, layers)
            all_err.append(np.linalg.norm(h - ref) / np.linalg.norm(ref))
            last_err.append(np.linalg.norm(h[-1] - ref[-1]) / np.linalg.norm(ref[-1]))
        print(f"{cand}: all tokens {np.mean(all_err):.4f}, answer position {np.mean(last_err):.4f} "
              f"(worst prompt {np.max(last_err):.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
