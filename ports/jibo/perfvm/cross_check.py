#!/usr/bin/env python3
"""cross_check.py DIR [--hidden-tol T] [--prob-tol P]: the gate's builds against one another.

Defaults 1e-5 for both: about eight times the largest difference measured between the builds
before any optimisation (1.2e-6 of the largest activation, 1.4e-6 in probability).

For every model DIR has a trace of from all three builds (x86-M, plain-M and neon-M.trace),
compares ARM plain and ARM NEON with x86: the same token ids and token counts, every layer's and
the final hidden states and the one-token continuations within T of the largest activation, every
answer probability within P, and the same choice for every question. Bit-identity per build
(golden.sha256) says an optimisation changed nothing; this says the builds still agree with one
another (x86 against the Python engine: python_check.py), so that a change cannot be bit-exact
on one build and wrong on another. Exit status 1 when a check fails."""
import argparse
import glob
import os
import re
import struct
import sys

import numpy as np


def sections(path):
    data = open(path, "rb").read()
    out, pos = {}, 0
    while pos < len(data):
        name = data[pos:pos + 16].rstrip(b"\0").decode()
        (n,) = struct.unpack_from("<I", data, pos + 16)
        out[name] = data[pos + 20:pos + 20 + n]
        pos += 20 + n
    return out


def compare(ref, got, hidden_tol, prob_tol):
    """(worst relative hidden difference, worst probability difference, failures)"""
    fails, wh, wp = [], 0.0, 0.0
    if ref["ids"] != got["ids"]:
        fails.append("token ids differ")
    for name in ref:
        if name == "ids":
            continue
        x, y = np.frombuffer(ref[name], np.float32), np.frombuffer(got[name], np.float32)
        if name.startswith("probs"):
            if x[-1] != y[-1]:
                fails.append(f"{name}: token count {y[-1]:.0f} against {x[-1]:.0f}")
            d = float(np.abs(x[:-1] - y[:-1]).max())
            wp = max(wp, d)
            if d > prob_tol:
                fails.append(f"{name}: probability differs by {d:.2e}")
        else:
            d = float(np.abs(x - y).max() / max(1e-30, float(np.abs(x).max())))
            wh = max(wh, d)
            if d > hidden_tol:
                fails.append(f"{name}: hidden states differ by {d:.2e} of the largest")
    return wh, wp, fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--hidden-tol", type=float, default=1e-5)
    ap.add_argument("--prob-tol", type=float, default=1e-5)
    args = ap.parse_args()
    models = sorted({re.sub(r"^x86-(.*)\.trace$", r"\1", os.path.basename(p))
                     for p in glob.glob(os.path.join(args.dir, "x86-*.trace"))})
    bad = 0
    for m in models:
        paths = {b: os.path.join(args.dir, f"{b}-{m}.trace") for b in ("x86", "plain", "neon")}
        if not all(os.path.exists(p) for p in paths.values()):
            continue
        ref = sections(paths["x86"])
        for b in ("plain", "neon"):
            wh, wp, fails = compare(ref, sections(paths[b]), args.hidden_tol, args.prob_tol)
            print(f"{m:8} {b:5} vs x86: hidden {wh:.2e} of the largest, probabilities {wp:.2e}"
                  + ("" if not fails else "  FAIL: " + "; ".join(fails[:3])))
            bad += bool(fails)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
