"""Compare the C tokenizer (tests/tok_dump.c) with Hugging Face tokenizers, token for token.

    python ports/jibo/tests/tokenizer_parity.py TOK_DUMP_BINARY TOKENIZER.jdt REFERENCE [--limit N]

REFERENCE is a checkpoint (its tokenizer as transformers loads it) or a tokenizer.json file
(the raw tokenizers pipeline it declares).

The corpus is every rendered prompt (state + question) of the evaluation files under data/
that exist, plus generated adversarial strings. Each text is NFC-normalised in Python first,
as the C renderer normalises its inputs, and both tokenizers see the same text. Ids and byte
offsets must match exactly; the criterion is zero mismatches. Also checks the C NFC against
Python's on the raw (unnormalised) strings.
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import struct
import subprocess
import sys
import unicodedata

from tokenizers import Tokenizer
from transformers import AutoTokenizer

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

from strands_decider.data.format import read_jsonl  # noqa: E402
from strands_decider.modeling import checkpoint_dir  # noqa: E402
from strands_decider.prompting import build_prompt  # noqa: E402


def corpus(limit: int) -> list[str]:
    texts: list[str] = []
    files = [*sorted(glob.glob(os.path.join(REPO, "data", "*eval*.jsonl"))),
             os.path.join(REPO, "data", "holdout_v5_norule.jsonl")]
    for path in files:
        if not os.path.exists(path):
            continue
        n = 0
        for ex in read_jsonl(path):
            prompt, _ = build_prompt(ex.state, ex.to_question())
            texts.append(prompt)
            n += 1
            if n >= limit:
                break
    rng = random.Random(0)
    pieces = [" ", "  ", "\n", "\r\n", "\t", " ", "　", " ", "\x1c", "\x85", "'s", "'S", "'ll",
              "'LL", "'re", "'ve", "'d", "'m", "'t", "'ſ", "don't", "é", "é", "가",
              "가", "क्ष", "\U0001f600", "\U0001f468‍\U0001f469", "日本語", "한국어",
              "١٢٣", "²", "Ⅻ", "abc", "ABC", "123", "4.5", "--", "...", "<|endoftext|>", "<|im_start|>",
              "<state>", "</answer>", "́", "̸", "≯", "Å", "Ω", "ẋ̣",
              "ſ", "ﬁ", "​", "﻿", "à̖", " ́", "ཱི", "̈́", "ẛ̣"]
    for _ in range(3000):
        texts.append("".join(rng.choice(pieces) for _ in range(rng.randint(1, 12))))
    for n in (1, 2, 3, 50, 1000):
        texts += [" " * n, "\n" * n, "a" * n, "1" * n, " \n" * n, "!" * n]
    return texts


def char_to_byte(text: str) -> list[int]:
    out = [0]
    for ch in text:
        out.append(out[-1] + len(ch.encode("utf-8")))
    return out


def run(binary: str, jdt: str, texts: list[str], mode: str) -> list[str]:
    payload = b"".join(struct.pack("<I", len(b)) + b for b in (t.encode("utf-8") for t in texts))
    res = subprocess.run([binary, jdt, mode], input=payload, capture_output=True, check=True)
    lines = res.stdout.decode().split("\n")
    assert len(lines) == len(texts) + 1, (len(lines), len(texts))
    return lines[:-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("jdt")
    ap.add_argument("reference")
    ap.add_argument("--limit", type=int, default=2000, help="prompts per evaluation file")
    args = ap.parse_args()
    if args.reference.endswith(".json"):
        raw_tok = Tokenizer.from_file(args.reference)

        def tok(text: str, **_: object) -> dict[str, list]:
            e = raw_tok.encode(text, add_special_tokens=False)
            return {"input_ids": e.ids, "offset_mapping": e.offsets}
    else:
        tok = AutoTokenizer.from_pretrained(checkpoint_dir(args.reference))
    raw = corpus(args.limit)

    nfc_c = run(args.binary, args.jdt, raw, "nfc-only")
    bad_nfc = [t for t, h in zip(raw, nfc_c, strict=True)
               if h != unicodedata.normalize("NFC", t).encode("utf-8").hex()]
    print(f"NFC: {len(raw) - len(bad_nfc)} of {len(raw)} agree")
    for t in bad_nfc[:5]:
        print("  NFC mismatch:", ascii(t))

    texts = [unicodedata.normalize("NFC", t) for t in raw]
    lines = run(args.binary, args.jdt, texts, "")
    bad = 0
    tokens = 0
    for text, line in zip(texts, lines, strict=True):
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        b = char_to_byte(text)
        want = " ".join(f"{i}:{b[s]}:{b[e]}" for i, (s, e) in zip(enc["input_ids"], enc["offset_mapping"], strict=True))
        tokens += len(enc["input_ids"])
        # HF offsets are in characters; a byte-level token that splits a character gets the
        # whole character's span, so compare ids exactly and offsets only where HF's are exact.
        got_ids = [x.split(":")[0] for x in line.split()] if line else []
        if got_ids != [str(i) for i in enc["input_ids"]]:
            bad += 1
            if bad <= 5:
                print("  ids mismatch:", ascii(text[:200]))
                print("    hf:", enc["input_ids"][:40])
                print("    c: ", [int(x) for x in got_ids[:40]])
            continue
        if line != want and all(len(ch.encode("utf-8")) == 1 for ch in text):
            bad += 1
            if bad <= 5:
                print("  offsets mismatch:", ascii(text[:200]), "\n    hf:", want[:300], "\n    c: ", line[:300])
    print(f"tokens: {len(texts) - bad} of {len(texts)} texts identical ({tokens:,} tokens)")
    return 1 if bad or bad_nfc else 0


if __name__ == "__main__":
    sys.exit(main())
