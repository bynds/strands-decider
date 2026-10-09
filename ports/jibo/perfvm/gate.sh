#!/usr/bin/env bash
# gate.sh: the bit-parity gate every optimisation passes before it is committed.
#
# Builds tests/trace.c from this tree three ways (x86 as the host builds it, ARMv7 plain
# VFPv3-D16 and ARMv7 NEON), runs each on the gate models (ARM under qemu-arm with the glibc 2.21
# sysroot), and requires every trace to hash to perfvm/golden.sha256: the same bytes, layer by
# layer, as the tree before any optimisation. Then the builds against one another
# (cross_check.py: every hidden state and probability within 1e-5) and, when the checkpoint is
# there, against the original Python engine (python_check.py, on the f32 four-layer model: within
# 1e-4 of the largest activation and 2e-5 in probability, the same choices). x86 runs every model; the ARM builds run the two
# four-layer ones (both layer types, q4, q8 and f32 matrices), since qemu-arm takes about an hour
# per build for the 24-layer model; GATE_FULL_ARM=1 adds it. Then the runtime tests on x86 and under qemu-arm
# (both FPUs), an AddressSanitizer + UBSan build of the trace on x86, and ruff.
#
#   JIBO_SYSROOT   the stand-in sysroot (scripts/fetch-sysroot.sh)
#   GATE_MODELS    name=DIR pairs (default: bench4, f32-4 and full under $TMPDIR/perfvm)
#   GATE_ARM_MODELS  the names the ARM builds run (default: bench4 f32-4)
#   GATE_OUT       where traces and logs go (default $TMPDIR/perfvm/gate)
#   GATE_RECORD=1  write golden.sha256 from this tree instead of checking against it
#   GATE_SKIP_TESTS=1  traces and sanitizers only
#   GATE_CHECKPOINT  the v22 checkpoint for the Python comparison (default: checkpoints/hobson-0.8b-v22
#                  in the repository; skipped, and said so, when absent)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
REPO=$(cd "$ROOT/../.." && pwd)
: "${JIBO_SYSROOT:?set JIBO_SYSROOT}"
W=${TMPDIR:-/tmp}/perfvm
OUT=${GATE_OUT:-$W/gate}
MODELS=${GATE_MODELS:-"bench4=$W/bench4 f32-4=$W/f32-4 full=$W/full"}
F=$HERE/fixtures
rm -rf "$OUT"
mkdir -p "$OUT"
fail() { echo "GATE FAIL: $*" >&2; exit 1; }

make -s -C "$ROOT" OUT="$OUT/x86" "$OUT/x86/trace"
make -s -C "$ROOT" OUT="$OUT/plain" CC="$ROOT/scripts/jibo-cc.sh" CFLAGS="-O2 -mfpu=vfpv3-d16" "$OUT/plain/trace"
make -s -C "$ROOT" OUT="$OUT/neon" CC="$ROOT/scripts/jibo-cc.sh" CFLAGS="-O2 -mfpu=neon" "$OUT/neon/trace"
make -s -C "$ROOT" OUT="$OUT/asan" CFLAGS="-O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all" \
  LDFLAGS="-fsanitize=address,undefined" "$OUT/asan/trace"

run() {  # run BUILD MODEL DIR [RUNNER...]
  local b=$1 m=$2 d=$3
  shift 3
  "$@" "$OUT/$b/trace" "$d/model.jdw" "$d/tokenizer.jdt" "$F/request1.json" "$F/request3.json" \
    "$OUT/$b-$m.trace" > "$OUT/$b-$m.log" 2>&1 || { cat "$OUT/$b-$m.log"; fail "$b on $m did not run"; }
}
ARM_MODELS=${GATE_ARM_MODELS:-bench4 f32-4}
[ -n "${GATE_FULL_ARM:-}" ] && ARM_MODELS="$ARM_MODELS full"
pids=()
for pair in $MODELS; do
  m=${pair%%=*}
  d=${pair#*=}
  run x86 "$m" "$d" & pids+=($!)
  case " $ARM_MODELS " in
    *" $m "*) run plain "$m" "$d" qemu-arm -L "$JIBO_SYSROOT" & pids+=($!)
              run neon "$m" "$d" qemu-arm -L "$JIBO_SYSROOT" & pids+=($!) ;;
  esac
done
for p in "${pids[@]}"; do wait "$p" || fail "a trace run failed"; done
run asan bench4 "$W/bench4" env ASAN_OPTIONS=detect_leaks=0

(cd "$OUT" && sha256sum -- *-*.trace | grep -v ' asan-' | sort -k2) > "$OUT/traces.sha256"
if [ -n "${GATE_RECORD:-}" ]; then
  cp "$OUT/traces.sha256" "$HERE/golden.sha256"
  echo "recorded $(wc -l < "$HERE/golden.sha256") golden traces -> $HERE/golden.sha256"
else
  join -j 2 <(sort -k2 "$HERE/golden.sha256") "$OUT/traces.sha256" | awk '$2 != $3' > "$OUT/traces.diff"
  orphans=$(join -j 2 -v 2 <(sort -k2 "$HERE/golden.sha256") "$OUT/traces.sha256")
  [ -z "$orphans" ] || fail "traces with no golden hash: $orphans"
  [ ! -s "$OUT/traces.diff" ] ||
    { cat "$OUT/traces.diff"; fail "traces differ from the golden ones (perfvm/trace_diff.py locates the change)"; }
  echo "traces: $(wc -l < "$OUT/traces.sha256") bit-identical to golden"
  cmp -s "$OUT/asan-bench4.trace" "$OUT/x86-bench4.trace" ||
    echo "note: the sanitizer build's trace differs from the -O2 x86 build's (expected only if -O1 changes codegen)"
fi
echo "sanitizers: clean"

python3 "$HERE/cross_check.py" "$OUT" > "$OUT/cross.log" || { cat "$OUT/cross.log"; fail "the builds disagree"; }
echo "builds: x86, ARM plain and ARM NEON agree within 1e-5 on $(cut -c1-8 "$OUT/cross.log" | sort -u | wc -l) models"
CKPT=${GATE_CHECKPOINT:-$REPO/checkpoints/hobson-0.8b-v22}
if [ -d "$CKPT" ] && [ -f "$OUT/x86-f32-4.trace" ]; then
  python3 "$HERE/python_check.py" "$CKPT" "$OUT/x86-f32-4.trace" "$OUT/plain-f32-4.trace" "$OUT/neon-f32-4.trace" \
    --layers 4 --hidden-tol 1e-4 --prob-tol 2e-5 > "$OUT/python.log" 2>&1 ||
    { grep "trace:" "$OUT/python.log" || tail -5 "$OUT/python.log"; fail "a build departs from the Python engine"; }
  echo "python: all three builds match the Python engine (f32, four layers): $(grep -c 'trace:' "$OUT/python.log") traces"
else
  echo "python: skipped (no checkpoint at $CKPT)"
fi

if [ -z "${GATE_SKIP_TESTS:-}" ]; then
  (cd "$REPO" && python -m pytest -q tests/test_jibo_runtime.py) > "$OUT/pytest-x86.log" 2>&1 ||
    { tail -20 "$OUT/pytest-x86.log"; fail "runtime tests (x86)"; }
  for fpu in vfpv3-d16 neon; do
    (cd "$REPO" && JIBO_CC="$ROOT/scripts/jibo-cc.sh" JIBO_CFLAGS="-O2 -mfpu=$fpu" \
      JIBO_RUNNER="qemu-arm -L $JIBO_SYSROOT" python -m pytest -q tests/test_jibo_runtime.py) \
      > "$OUT/pytest-$fpu.log" 2>&1 || { tail -20 "$OUT/pytest-$fpu.log"; fail "runtime tests ($fpu, qemu-arm)"; }
  done
  echo "runtime tests: x86, ARM plain, ARM NEON pass"
  (cd "$REPO" && ruff check -q .) || fail "ruff"
fi
echo "GATE PASS"
