r"""Write a checkpoint's tokenizer as a .jdt file for the C runtime (runtime/jd_tokenizer.c).

Layout, little-endian:

    header (32 bytes)   "JDT1", u32 version (1), u32 n_vocab, u32 blob_len, u32 hash_slots,
                        u32 n_added, u32 flags (bit 0: the split counts \p{M} with letters),
                        4 reserved bytes
    byte_id             i32[256], the token id of each single byte
    vocab_off           u32[n_vocab + 1], offsets of each token's raw bytes in the blob
    blob                the raw bytes of every token, padded to 16
    hash                hash_slots x {u32 a, u32 b, u32 rank, u32 result}, open addressing
                        (merge_hash, linear probing); a == 0xFFFFFFFF marks an empty slot
    added tokens        n_added x {u32 id, u32 len, u8 special, normalized, lstrip, rstrip,
                        bytes padded to 4}

Only the tokenizer pipeline the C code implements is accepted: NFC, one of the two Qwen split
regexes, byte-level with no prefix space, plain BPE. Anything else is refused rather than
ignored. The pipeline is read from the tokenizer as transformers loads it, not from the
checkpoint's tokenizer.json: Qwen2Tokenizer substitutes its own split regex for the file's.
"""

from __future__ import annotations

import json
import struct
from typing import Any

# Qwen2Tokenizer's split, and Qwen3.5's own (marks counted with letters). The flag records which.
SPLITS = {
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+|\s+(?!\S)|\s+": 0,
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+|\s+(?!\S)|\s+": 1,
}
EMPTY = 0xFFFFFFFF


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's byte-level alphabet: each byte as a printable character."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(
        range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs, strict=True)}


def merge_hash(a: int, b: int, slots: int) -> int:
    """Must match merge_hash() in runtime/jd_tokenizer.c."""
    return (((a * 0x9E3779B1) & 0xFFFFFFFF) ^ ((b * 0x85EBCA77) & 0xFFFFFFFF)) & (slots - 1)


def check_pipeline(tj: dict[str, Any]) -> int:
    """Refuse a pipeline the C tokenizer does not implement; return the split flag."""
    def need(cond: bool, what: str) -> None:
        if not cond:
            raise ValueError(f"tokenizer not supported by the C runtime: {what}")

    need(tj.get("normalizer") == {"type": "NFC"}, f"normalizer {tj.get('normalizer')}")
    pt = tj.get("pre_tokenizer") or {}
    seq = pt.get("pretokenizers") or []
    need(pt.get("type") == "Sequence" and len(seq) == 2, "pre_tokenizer is not Split + ByteLevel")
    split, bl = seq
    pattern = (split.get("pattern") or {}).get("Regex")
    need(split.get("type") == "Split" and pattern in SPLITS
         and split.get("behavior") == "Isolated" and not split.get("invert"),
         f"split {split}")
    need(bl.get("type") == "ByteLevel" and not bl.get("add_prefix_space") and not bl.get("use_regex"),
         f"byte level {bl}")
    pp = tj.get("post_processor")
    # A ByteLevel post-processor only trims offsets, and only with trim_offsets.
    need(pp is None or (pp.get("type") == "ByteLevel" and not pp.get("trim_offsets")),
         f"post_processor {pp}")
    m = tj["model"]
    need(m.get("type") == "BPE" and not m.get("ignore_merges") and not m.get("byte_fallback")
         and not m.get("continuing_subword_prefix") and not m.get("end_of_word_suffix")
         and m.get("dropout") is None, "model is not plain BPE")
    return SPLITS[pattern]


def build_jdt(tokenizer_json: str) -> bytes:
    """tokenizer_json: the backend tokenizer serialised, `tok.backend_tokenizer.to_str()`."""
    tj = json.loads(tokenizer_json)
    flags = check_pipeline(tj)
    dec = {c: b for b, c in bytes_to_unicode().items()}
    vocab: dict[str, int] = tj["model"]["vocab"]
    added = tj.get("added_tokens", [])
    for a in added:
        if a.get("normalized") or a.get("lstrip") or a.get("rstrip") or a.get("single_word"):
            raise ValueError(f"added token {a['content']!r} needs matching the C runtime lacks")
    n_vocab = max([*vocab.values(), *(a["id"] for a in added)]) + 1
    raw: list[bytes] = [b""] * n_vocab
    for tok, i in vocab.items():
        raw[i] = bytes(dec[ch] for ch in tok)
    for a in added:
        raw[a["id"]] = a["content"].encode("utf-8")
    byte_id = [vocab[bytes_to_unicode()[b]] for b in range(256)]

    merges = tj["model"]["merges"]
    pairs = [m.split(" ") if isinstance(m, str) else m for m in merges]
    slots = 1
    while slots < 2 * len(pairs):
        slots *= 2
    table = [(EMPTY, EMPTY, EMPTY, EMPTY)] * slots
    for rank, (a, b) in enumerate(pairs):
        ia, ib, ir = vocab[a], vocab[b], vocab[a + b]
        h = merge_hash(ia, ib, slots)
        while table[h][0] != EMPTY:
            if table[h][:2] == (ia, ib):
                break  # a repeated merge keeps its first (lowest) rank
            h = (h + 1) & (slots - 1)
        else:
            table[h] = (ia, ib, rank, ir)

    blob = b"".join(raw)
    offs = [0]
    for r in raw:
        offs.append(offs[-1] + len(r))
    out = bytearray()
    out += b"JDT1" + struct.pack("<6I", 1, n_vocab, len(blob), slots, len(added), flags) + b"\0" * 4
    out += struct.pack(f"<{256}i", *byte_id)
    out += struct.pack(f"<{n_vocab + 1}I", *offs)
    out += blob + b"\0" * ((-len(blob)) % 16)
    out += b"".join(struct.pack("<4I", *s) for s in table)
    for a in added:
        text = a["content"].encode("utf-8")
        out += struct.pack("<2I4B", a["id"], len(text), int(a.get("special", False)), 0, 0, 0)
        out += text + b"\0" * ((-len(text)) % 4)
    return bytes(out)
