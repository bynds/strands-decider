"""Write a decider checkpoint as a .jdw file for the C runtime (runtime/jd_model.c).

Layout, little-endian:

    header (24 bytes)   "JDW1", u32 version (1), u32 meta_len, u32 n_tensors, u64 data_offset
    meta                meta_len bytes of UTF-8 "key=value" lines
    tensor table        n_tensors x {u16 name_len, name, u8 dtype, u8 ndim, u32 dims[ndim],
                        u64 offset (absolute), u64 nbytes}
    data                each tensor at a 64-byte boundary

dtypes: 0 f32, 1 bf16, 3 q8, 4 q4. A quantised matrix [rows, cols] is stored row by row in
blocks of 32 columns: q8 is {f16 scale, 32 x i8} (34 bytes), q4 is {f16 scale, 16 bytes of
nibbles, low nibble first} (18 bytes), value = scale * q (q4 stores q + 8).

The LoRA adapter is folded in fp32 before anything is rounded, W + (alpha / r) B A, which is
what PEFT's merge computes. The Qwen3.5 RMSNorm weights are stored as (1 + w), the factor the
model multiplies by. Neither the vision tower nor the multi-token-prediction layer is written.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import torch

F32, BF16, Q8, Q4 = 0, 1, 3, 4
QBLOCK = 32
MATRIX_DTYPES = {"f32": F32, "q8": Q8, "q4": Q4}


@dataclass
class Tensor:
    """A tensor to write. `make` produces its bytes when the writer reaches it, so the file
    is streamed and never held whole in memory."""

    name: str
    dtype: int
    dims: tuple[int, ...]
    make: Callable[[], bytes]

    @property
    def nbytes(self) -> int:
        n = math.prod(self.dims)
        if self.dtype == F32:
            return 4 * n
        if self.dtype == BF16:
            return 2 * n
        return n // QBLOCK * (34 if self.dtype == Q8 else 18)


def quantise(w: np.ndarray, dtype: int) -> bytes:
    """Group-of-32 symmetric quantisation along each row, as runtime/jd_quant.c dequantises."""
    rows, cols = w.shape
    if cols % QBLOCK:
        raise ValueError(f"{cols} columns is not a multiple of {QBLOCK}")
    g = w.reshape(rows, cols // QBLOCK, QBLOCK).astype(np.float32)
    amax = np.abs(g).max(axis=-1, keepdims=True)
    qmax = 127.0 if dtype == Q8 else 7.0
    lo = -127.0 if dtype == Q8 else -8.0
    # The scale of each block: amax / qmax shrunk by the factor (of a few) with the least squared
    # error after rounding. Clipping the largest value a little can make every other step finer;
    # it matters at 4 bits. The stored format is the same either way.
    best_err = np.full(amax.shape, np.inf, dtype=np.float32)
    scale16 = (amax / qmax).astype(np.float16)
    for f in ((1.0,) if dtype == Q8 else np.linspace(1.0, 0.8, 11)):
        cand16 = (amax * f / qmax).astype(np.float16)
        c = cand16.astype(np.float32)
        ic = np.divide(1.0, c, out=np.zeros_like(c), where=c > 0)
        err = ((np.clip(np.rint(g * ic), lo, qmax) * c - g) ** 2).sum(axis=-1, keepdims=True)
        better = err < best_err
        scale16 = np.where(better, cand16, scale16)
        best_err = np.where(better, err, best_err)
    scale = scale16.astype(np.float32)
    inv = np.divide(1.0, scale, out=np.zeros_like(scale), where=scale > 0)
    if dtype == Q8:
        q = np.clip(np.rint(g * inv), -127, 127).astype(np.int8)
        blocks = np.concatenate([scale16.view(np.uint8).reshape(rows, -1, 2),
                                 q.view(np.uint8)], axis=-1)
    else:
        q = (np.clip(np.rint(g * inv), -8, 7) + 8).astype(np.uint8)
        packed = (q[..., 0::2] | (q[..., 1::2] << 4)).astype(np.uint8)
        blocks = np.concatenate([scale16.view(np.uint8).reshape(rows, -1, 2), packed], axis=-1)
    return np.ascontiguousarray(blocks).tobytes()


def dequantise(data: bytes, dtype: int, rows: int, cols: int) -> np.ndarray:
    """The inverse, for tests: what the C runtime computes from the bytes."""
    nb = cols // QBLOCK
    width = 34 if dtype == Q8 else 18
    b = np.frombuffer(data, dtype=np.uint8).reshape(rows, nb, width)
    scale = b[..., :2].copy().view(np.float16).astype(np.float32)
    if dtype == Q8:
        q = b[..., 2:].copy().view(np.int8).astype(np.float32)
    else:
        p = b[..., 2:]
        q = np.empty((rows, nb, QBLOCK), dtype=np.float32)
        q[..., 0::2] = (p & 15).astype(np.float32) - 8
        q[..., 1::2] = (p >> 4).astype(np.float32) - 8
    return (q * scale).reshape(rows, cols)


def _np32(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float32).contiguous().numpy()


def f32(name: str, t: torch.Tensor) -> Tensor:
    return Tensor(name, F32, tuple(t.shape), lambda: _np32(t).tobytes())


def matrix(name: str, t: torch.Tensor, kind: str) -> Tensor:
    if kind == "f32":
        return f32(name, t)
    dt = MATRIX_DTYPES[kind]
    return Tensor(name, dt, tuple(t.shape), lambda: quantise(_np32(t), dt))


def bf16(name: str, t: torch.Tensor) -> Tensor:
    return Tensor(name, BF16, tuple(t.shape),
                  lambda: t.detach().to(torch.bfloat16).contiguous().view(torch.int16).numpy().tobytes())


def tensors_from_model(model: torch.nn.Module, head: torch.nn.Module, cfg: object, kind: str,
                       embed_kind: str) -> Iterable[Tensor]:
    """The torso (LoRA already merged, fp32) and the pointer head, in runtime names."""
    emb = model.embed_tokens.weight
    if embed_kind == "bf16":
        yield bf16("embed", emb)
    else:
        yield f32("embed", emb)
    for i, layer in enumerate(model.layers):
        p = f"l{i}."
        yield f32(p + "in_norm", 1.0 + layer.input_layernorm.weight.float())
        yield f32(p + "post_norm", 1.0 + layer.post_attention_layernorm.weight.float())
        if layer.block_type == "linear_attention":
            la = layer.linear_attn
            yield matrix(p + "qkv", la.in_proj_qkv.weight, kind)
            yield matrix(p + "z", la.in_proj_z.weight, kind)
            yield f32(p + "b", la.in_proj_b.weight)  # 16 rows: kept fp32
            yield f32(p + "a", la.in_proj_a.weight)
            yield f32(p + "conv", la.conv1d.weight.squeeze(1))
            yield f32(p + "dt_bias", la.dt_bias)
            yield f32(p + "a_log", la.A_log)
            yield f32(p + "gnorm", la.norm.weight)
            yield matrix(p + "out", la.out_proj.weight, kind)
        else:
            at = layer.self_attn
            yield matrix(p + "q", at.q_proj.weight, kind)
            yield matrix(p + "k", at.k_proj.weight, kind)
            yield matrix(p + "v", at.v_proj.weight, kind)
            yield matrix(p + "o", at.o_proj.weight, kind)
            yield f32(p + "q_norm", 1.0 + at.q_norm.weight.float())
            yield f32(p + "k_norm", 1.0 + at.k_norm.weight.float())
        yield matrix(p + "gate", layer.mlp.gate_proj.weight, kind)
        yield matrix(p + "up", layer.mlp.up_proj.weight, kind)
        yield matrix(p + "down", layer.mlp.down_proj.weight, kind)
    yield f32("final_norm", 1.0 + model.norm.weight.float())
    yield f32("head.ln_w", head.norm.weight)
    yield f32("head.ln_b", head.norm.bias)
    yield f32("head.q_w", head.q.weight)
    yield f32("head.q_b", head.q.bias)
    yield f32("head.k_w", head.k.weight)
    yield f32("head.k_b", head.k.bias)


def write_jdw(path: str, meta: dict[str, str], tensors: Iterable[Tensor]) -> str:
    """Write the file; returns its SHA-256."""
    ts = list(tensors)
    meta_b = "".join(f"{k}={v}\n" for k, v in meta.items()).encode("utf-8")
    table_len = sum(2 + len(t.name.encode()) + 2 + 4 * len(t.dims) + 16 for t in ts)
    off = 24 + len(meta_b) + table_len
    data_offset = (off + 63) & ~63
    table = bytearray()
    pos = data_offset
    placed = []
    for t in ts:
        name = t.name.encode()
        table += struct.pack("<H", len(name)) + name + struct.pack("<BB", t.dtype, len(t.dims))
        table += struct.pack(f"<{len(t.dims)}I", *t.dims) + struct.pack("<QQ", pos, t.nbytes)
        placed.append(pos)
        pos = (pos + t.nbytes + 63) & ~63
    h = hashlib.sha256()
    with open(path, "wb") as fh:
        head = b"JDW1" + struct.pack("<IIIQ", 1, len(meta_b), len(ts), data_offset)
        for chunk in (head, meta_b, bytes(table), b"\0" * (data_offset - off)):
            fh.write(chunk)
            h.update(chunk)
        cur = data_offset
        for t, p in zip(ts, placed, strict=True):
            pad = b"\0" * (p - cur)
            fh.write(pad)
            h.update(pad)
            data = t.make()
            if len(data) != t.nbytes:
                raise AssertionError(f"{t.name}: {len(data)} bytes, expected {t.nbytes}")
            fh.write(data)
            h.update(data)
            cur = p + len(data)
    return h.hexdigest()
