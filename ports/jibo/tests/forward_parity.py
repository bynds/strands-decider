"""Compare the C torso (tests/fwd_dump.c) with the torch engine's, layer by layer.

    python ports/jibo/tests/forward_parity.py FWD_DUMP MODEL.jdw CHECKPOINT [--chunk N] [--split K]

The prompt is rendered and tokenised by the Python engine; both sides run the same ids.
The torch side is the serving path on CPU: the torso in fp32 with the LoRA adapter unmerged.
Reports, for every layer, the largest absolute difference relative to that layer's largest
activation, and fails above --tol.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

from strands_decider.infer import load_engine  # noqa: E402
from strands_decider.prompting import build_prompt  # noqa: E402
from strands_decider.schema import ChoiceQuestion  # noqa: E402

STATE = ("Refunds and payout disputes. A customer may request a refund within 30 days of the "
         "charge date. My payouts have failed for three days and I have rent due tomorrow. "
         "This is unacceptable. I already updated my bank details twice.")
QUESTION = ChoiceQuestion(
    instructions="Which team should handle this ticket?",
    criteria={"billing": "charges, refunds, payouts", "technical": "bugs and outages",
              "account": "identity and settings", "sales": "new purchases"})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("jdw")
    ap.add_argument("checkpoint")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--split", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()

    eng = load_engine(args.checkpoint, device="cpu")
    prompt, _ = build_prompt(STATE, QUESTION)
    ids = eng.tok(prompt, add_special_tokens=True)["input_ids"]
    n = len(ids)
    with tempfile.TemporaryDirectory() as d:
        np.asarray(ids, dtype=np.int32).tofile(os.path.join(d, "ids.bin"))
        subprocess.run([args.binary, args.jdw, os.path.join(d, "ids.bin"), os.path.join(d, "out.bin"),
                        str(args.chunk), str(args.split)], check=True)
        got = np.fromfile(os.path.join(d, "out.bin"), dtype=np.float32)

    torso = eng.model.torso
    layers = torso.base_model.model.layers if hasattr(torso, "base_model") else torso.layers
    outs: list[torch.Tensor] = []
    hooks = [lyr.register_forward_hook(lambda _m, _i, o: outs.append(o[0] if isinstance(o, tuple) else o))
             for lyr in layers]
    with torch.inference_mode():
        last = torso(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, n, dtype=torch.long),
                     return_dict=True).last_hidden_state
    for h in hooks:
        h.remove()
    want = [o[0].float().numpy() for o in outs] + [last[0].float().numpy()]
    hidden = want[0].shape[-1]
    got = got.reshape(len(want), n, hidden)
    worst = 0.0
    for i, w in enumerate(want):
        rel = float(np.abs(got[i] - w).max() / max(1e-12, np.abs(w).max()))
        worst = max(worst, rel)
        name = f"layer {i:2d}" if i < len(want) - 1 else "final   "
        print(f"{name}  max |diff| / max |x| = {rel:.2e}   max |x| = {np.abs(w).max():.3g}")
    print(f"{n} tokens, chunk {args.chunk}, split {args.split}: worst {worst:.2e} (tol {args.tol:g})")
    return 0 if worst <= args.tol else 1


if __name__ == "__main__":
    sys.exit(main())
