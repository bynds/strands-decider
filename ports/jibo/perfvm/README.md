# perfvm: exact ARMv7 instruction counts, and the bit-parity gate

The runtime's cost on Jibo's instruction set, measured without a robot, and the gate every
optimisation passes. Adapted from the Needle port's perfvm (bynds/needle-rs, `ports/jibo/perfvm`).

## Counting

`qemu-user`, which the tests use, emulates fine but has no PMU: a guest's `perf_event_open` would
count the emulator's own x86 instructions. Full-system qemu does have one. With `-icount shift=0`,
TCG retires instructions deterministically, and the emulated PMU's instructions-retired event is
exact. The guest is an arm64 Ubuntu kernel (`fetch-kernel.sh`, pinned by sha256) on a Cortex-A57.
It runs the unmodified armhf binaries in AArch32 EL0 against the glibc 2.21 stand-in sysroot
(`scripts/fetch-sysroot.sh`), from an initramfs that `mkinitramfs.py` writes with no root access.
`init.c` runs a tab-separated jobs file with `LD_BIND_NOW=1`, so that lazy symbol binding adds
nothing, and powers off. `selftest.c` checks the counter: a 6-instruction loop costs 6 per
iteration, and the same on every run.

```sh
export JIBO_SYSROOT=/path/to/sysroot TMPDIR=/path/to/scratch
perfvm/bench.sh LABEL                          # build, run, perfvm/results/LABEL.jsonl
perfvm/compare.py results/NEW.jsonl results/OLD.jsonl   # per-operation table and change
perfvm/run.sh JOBS DEST=SRC...                 # any ARMv7 programs
```

`bench.sh` builds `bench_engine.c` with `-DJD_PROFILE` twice, plain (`-mfpu=vfpv3-d16`, what
ships) and NEON, and runs three workloads on the fixtures' requests:

- **prefill:** `request1.json` through `jd_evaluate`, rendering and tokenising included: one forward
  of its 109-token prompt (state and question) and the pointer head;
- **prefix_hit:** after `request3.json`, copying the cached state prefix and forwarding one
  question (68 tokens) on it, which each question of a multi-question request costs;
- **request3:** `request3.json` end to end: the 44-token state once, then three questions on it.

Each line of the JSONL gives the total and, per operation of `runtime/jd_prof.h`, the
instructions and the span count: tokenize, embed, lin_proj, lin_conv, lin_rec, lin_out (DeltaNet
layers), attn_proj, attn_rope, attn, attn_out (attention layers), mlp, head, state, and other.
Without `-DJD_PROFILE` the spans compile to nothing.

The model is the first four layers of the v22 export (`tools/truncate_jdw.py`, three DeltaNet
layers and one attention layer, the model's 3:1 ratio, with its q4 and q8 matrices): counting a
24-layer forward under icount would take most of an hour. Every layer of a type costs the same,
so the per-layer figures carry over; work done once per forward (the head, the final norm) is
over-weighted by about six.

What the counts are not: time. An instruction on a Cortex-A9 or A15 differs in cost (a NEON
`vmla`, a `vldm` of eight registers, a cache miss), and qemu models none of it. They are the
right measure for comparing two kernels that do the same work. Absolute latency needs the robot.

## The gate

`gate.sh` must pass before an optimisation is committed. It builds `tests/trace.c` from the tree
for x86, ARM plain and ARM NEON, and runs it on three models (`bench4`, `f32-4`: the same four
layers exported in f32, and `full`: the whole v22 dynamic-q4 export). A trace holds the prompt's
token ids, every layer's hidden states over it (in two chunks), the final hidden states, eight
one-token continuations, and the answers of `request1`, of `request3` with the shared prefix and
of `request3` without it, all as raw bytes. Each must hash to `golden.sha256`, which was recorded
from the tree before any optimisation. `trace_diff.py` says where two traces part.

The three builds do not agree with one another, and are not meant to: x86 links glibc 2.39's
`expf`, the ARM builds glibc 2.21's, and the NEON kernels sum in a different order from the plain
ones. Each build is held to its own golden: an optimisation may not move a single bit of any of
them. x86 runs all three models; the ARM builds run the four-layer ones under qemu-arm (the full
model takes about an hour per build there; `GATE_FULL_ARM=1` adds it). The gate then runs the
runtime tests on x86 and under qemu-arm for both FPUs, an AddressSanitizer and UBSan build of the
trace, and ruff.
