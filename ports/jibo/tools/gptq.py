"""GPTQ for the exporter: 4-bit weights in the same q4 format, with rounding error compensated.

Plain rounding treats every weight alone. GPTQ (Frantar et al., 2022) quantises a matrix one
input column at a time and spreads each column's rounding error over the columns not yet
quantised, weighted by the inverse Hessian H = 2 X^T X of the activations X that reach the
matrix. The stored format is unchanged: blocks of 32 inputs per row, an fp16 scale each, chosen
when the block is reached from the weights as updated so far. Only the exporter changes.

calibration_hessians() runs the merged fp32 torso over calibration prompts and accumulates H
for every matrix the exporter quantises; quantise_gptq() then quantises one matrix with it.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch

QBLOCK = 32
# The matrices of a layer that share an input share a Hessian.
SHARED_INPUT = {"qkv": "in_lin", "z": "in_lin", "q": "in_attn", "k": "in_attn", "v": "in_attn",
                "gate": "in_mlp", "up": "in_mlp", "out": "out", "o": "o", "down": "down"}


def calibration_hessians(torso: torch.nn.Module, prompts: Iterable[list[int]]) -> dict[str, torch.Tensor]:
    """{"l{i}.{input name}": sum over tokens of x x^T, and "count"} for every quantised matrix."""
    acc: dict[str, torch.Tensor] = {}
    count = {"n": 0}
    hooks = []

    def hook_for(key: str):
        def fn(_mod: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            # fp32 sums: fp64 Hessians of a 2B torso's down-projections alone would take 7 GB
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).to(torch.float32)
            h = x.T @ x
            acc[key] = acc[key] + h if key in acc else h
        return fn

    for i, layer in enumerate(torso.layers):
        p = f"l{i}."
        if layer.block_type == "linear_attention":
            la = layer.linear_attn
            hooks += [la.in_proj_qkv.register_forward_pre_hook(hook_for(p + "in_lin")),
                      la.out_proj.register_forward_pre_hook(hook_for(p + "out"))]
        else:
            at = layer.self_attn
            hooks += [at.q_proj.register_forward_pre_hook(hook_for(p + "in_attn")),
                      at.o_proj.register_forward_pre_hook(hook_for(p + "o"))]
        hooks += [layer.mlp.gate_proj.register_forward_pre_hook(hook_for(p + "in_mlp")),
                  layer.mlp.down_proj.register_forward_pre_hook(hook_for(p + "down"))]
    try:
        with torch.no_grad():
            for ids in prompts:
                torso(input_ids=torch.tensor([ids]), use_cache=False)
                count["n"] += len(ids)
    finally:
        for h in hooks:
            h.remove()
    for v in acc.values():
        v /= max(1, count["n"])
    return acc


def _scale(block: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """The fp16 scale per row of a block of 32 columns: amax / hi, shrunk by the factor with the
    least squared rounding error (as jdw.quantise chooses it)."""
    amax = np.abs(block).max(axis=1, keepdims=True)
    best = np.full(amax.shape, np.inf)
    out = (amax / hi).astype(np.float16)
    for f in np.linspace(1.0, 0.8, 11):
        c16 = (amax * f / hi).astype(np.float16)
        c = c16.astype(np.float64)
        ic = np.divide(1.0, c, out=np.zeros_like(c), where=c > 0)
        err = ((np.clip(np.rint(block * ic), lo, hi) * c - block) ** 2).sum(axis=1, keepdims=True)
        better = err < best
        out = np.where(better, c16, out)
        best = np.where(better, err, best)
    return out


def quantise_gptq(w: np.ndarray, hessian: np.ndarray, damp: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Quantise w [rows, cols] to 4 bits with GPTQ. Returns (q in -8..7 as int8 [rows, cols],
    fp16 scales [rows, cols / 32])."""
    rows, cols = w.shape
    W = w.astype(np.float64).copy()
    H = hessian.astype(np.float64).copy()
    dead = np.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    H += damp * np.mean(np.diag(H)) * np.eye(cols)
    # Upper Cholesky factor of H^-1: row i holds the update weights for columns after i.
    Hinv = np.linalg.cholesky(np.linalg.inv(H)).T
    q = np.zeros((rows, cols), dtype=np.int8)
    scales = np.zeros((rows, cols // QBLOCK), dtype=np.float16)
    lazy = 4 * QBLOCK  # columns whose errors are applied together, as one matmul, to the rest
    for c0 in range(0, cols, lazy):
        c1 = min(cols, c0 + lazy)
        W1 = W[:, c0:c1].copy()
        E1 = np.zeros_like(W1)
        Hi = Hinv[c0:c1, c0:c1]
        for j in range(c1 - c0):
            if j % QBLOCK == 0:  # a new block of 32: its scale from the weights as updated so far
                s16 = _scale(W1[:, j:j + QBLOCK], -8.0, 7.0)
                scales[:, (c0 + j) // QBLOCK] = s16[:, 0]
                s = s16.astype(np.float64)[:, 0]
                inv = np.divide(1.0, s, out=np.zeros_like(s), where=s > 0)
            qj = np.clip(np.rint(W1[:, j] * inv), -8, 7)
            q[:, c0 + j] = qj
            err = (W1[:, j] - qj * s) / Hi[j, j]
            W1[:, j + 1:] -= np.outer(err, Hi[j, j + 1:])
            E1[:, j] = err
        W[:, c1:] -= E1 @ Hinv[c0:c1, c1:]
    return q, scales


def pack_q4(q: np.ndarray, scales: np.ndarray) -> bytes:
    """The q4 block layout of jdw.py: per row, per 32 columns, the fp16 scale then 16 bytes of
    nibbles (value + 8), low nibble first."""
    rows, cols = q.shape
    u = (q.astype(np.int16) + 8).astype(np.uint8).reshape(rows, cols // QBLOCK, QBLOCK)
    packed = (u[..., 0::2] | (u[..., 1::2] << 4)).astype(np.uint8)
    blocks = np.concatenate([scales.reshape(rows, -1, 1).view(np.uint8).reshape(rows, -1, 2), packed], axis=-1)
    return np.ascontiguousarray(blocks).tobytes()


def hessian_error(w: np.ndarray, wq: np.ndarray, hessian: np.ndarray) -> float:
    """The quantisation error of a matrix as its output sees it on the calibration activations,
    relative to the output: tr(E H E^T) / tr(W H W^T), E = W - Wq. It is what GPTQ minimises,
    and a sensitivity score in the spirit of an importance matrix."""
    H = hessian.astype(np.float64)
    e = (w - wq).astype(np.float64)
    w64 = w.astype(np.float64)
    return float(np.einsum("ij,jk,ik->", e, H, e) / max(1e-30, np.einsum("ij,jk,ik->", w64, H, w64)))


def allocate_q8(scores: dict[str, float], extra_bytes: dict[str, int], budget: int) -> list[str]:
    """Which matrices to raise from q4 to q8 within `budget` extra bytes: greedily the largest
    error removed per byte spent (q8's own error is about 1% of q4's, so upgrading a matrix
    removes nearly all of its error)."""
    order = sorted(scores, key=lambda n: scores[n] / extra_bytes[n], reverse=True)
    chosen, spent = [], 0
    for name in order:
        if spent + extra_bytes[name] <= budget:
            chosen.append(name)
            spent += extra_bytes[name]
    return chosen
