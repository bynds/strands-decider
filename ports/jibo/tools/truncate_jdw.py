"""Keep the first N layers of a .jdw export: a smaller model with the real weights and shapes.

    python tools/truncate_jdw.py IN.jdw OUT.jdw N

The instruction-counting benchmarks (perfvm/bench.sh) run on a truncated export, because counting
a whole 24-layer forward under full-system emulation takes too long. Every layer of a type costs
the same, so the per-layer figures carry over; N = 4 keeps the model's 3:1 DeltaNet to attention
ratio (L, L, L, A). The answers of a truncated model mean nothing.
"""

from __future__ import annotations

import mmap
import re
import struct
import sys

from jdw import Tensor, write_jdw


def read_table(buf: bytes) -> tuple[dict[str, str], list[tuple[str, int, tuple[int, ...], int, int]]]:
    if buf[:4] != b"JDW1":
        raise ValueError("not a .jdw file")
    _, meta_len, n, _ = struct.unpack_from("<IIIQ", buf, 4)
    meta = dict(line.split("=", 1) for line in buf[24:24 + meta_len].decode().splitlines() if "=" in line)
    pos, out = 24 + meta_len, []
    for _ in range(n):
        (nl,) = struct.unpack_from("<H", buf, pos)
        name = buf[pos + 2:pos + 2 + nl].decode()
        pos += 2 + nl
        dtype, ndim = struct.unpack_from("<BB", buf, pos)
        dims = struct.unpack_from(f"<{ndim}I", buf, pos + 2)
        off, nbytes = struct.unpack_from("<QQ", buf, pos + 2 + 4 * ndim)
        pos += 2 + 4 * ndim + 16
        out.append((name, dtype, dims, off, nbytes))
    return meta, out


def main() -> int:
    src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    with open(src, "rb") as fh:
        buf = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        meta, table = read_table(buf)
        if n > int(meta["num_layers"]):
            raise ValueError(f"{src} has {meta['num_layers']} layers")
        meta["num_layers"] = str(n)
        meta["layer_types"] = ",".join(meta["layer_types"].split(",")[:n])
        meta["truncated_from"] = src
        keep = []
        for name, dtype, dims, off, nbytes in table:
            m = re.match(r"l(\d+)\.", name)
            if m and int(m.group(1)) >= n:
                continue
            keep.append(Tensor(name, dtype, tuple(dims), lambda off=off, nbytes=nbytes: bytes(buf[off:off + nbytes])))
        print(write_jdw(dst, meta, keep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
