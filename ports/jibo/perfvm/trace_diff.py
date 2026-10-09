#!/usr/bin/env python3
"""trace_diff.py A.trace B.trace: where two traces of tests/trace.c differ, section by section:
how many values, the largest absolute difference and the largest distance in units in the last
place. Exit status 1 when they differ."""
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


a, b = sections(sys.argv[1]), sections(sys.argv[2])
differ = 0
for name in a:
    if name not in b:
        print(f"{name}: only in {sys.argv[1]}")
        differ = 1
        continue
    if a[name] == b[name]:
        continue
    differ = 1
    if name == "ids":
        print("ids: different tokens")
        continue
    x, y = np.frombuffer(a[name], np.float32), np.frombuffer(b[name], np.float32)
    if x.shape != y.shape:
        print(f"{name}: {x.size} against {y.size} values")
        continue
    ix, iy = x.view(np.int32).astype(np.int64), y.view(np.int32).astype(np.int64)
    ix = np.where(ix < 0, -(ix & 0x7FFFFFFF), ix)
    iy = np.where(iy < 0, -(iy & 0x7FFFFFFF), iy)
    print(f"{name}: {int((x != y).sum())} of {x.size} values differ, max |diff| {float(np.abs(x - y).max()):.3g}, "
          f"max {int(np.abs(ix - iy).max())} ulp")
print("identical" if not differ else "")
sys.exit(differ)
