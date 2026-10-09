#!/usr/bin/env bash
# bench-full.sh LABEL [REF] [WORKLOAD]: count one workload (default prefill) on the full 24-layer
# model for the runtime at git REF (default HEAD), plain and NEON, in two perfvm guests side by
# side; writes $PERFVM_OUT/LABEL.jsonl. The bench driver is always this tree's, so a ref from
# before it existed is measured the same way.
#   JIBO_SYSROOT   the stand-in sysroot      FULL_MODEL   directory with the v22 model.jdw and
#   PERFVM_OUT     default perfvm/results                 tokenizer.jdt (default $TMPDIR/perfvm/full)
set -euo pipefail
LABEL=${1:?usage: bench-full.sh LABEL [REF] [WORKLOAD]}
REF=${2:-HEAD}
WL=${3:-prefill}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
REPO=$(cd "$ROOT/../.." && pwd)
: "${JIBO_SYSROOT:?set JIBO_SYSROOT}"
W=${TMPDIR:-/tmp}/perfvm
MODEL=${FULL_MODEL:-$W/full}
OUT=${PERFVM_OUT:-$HERE/results}
B=$W/full-$LABEL
rm -rf "$B"
mkdir -p "$B/src" "$OUT"
git -C "$REPO" archive "$REF" ports/jibo | tar -x -C "$B/src"
S=$B/src/ports/jibo
cp "$HERE/bench_engine.c" "$HERE/jd_icount.h" "$S/perfvm/"
grep -q '^\$(OUT)/bench-engine' "$S/Makefile" || cp "$ROOT/Makefile" "$S/Makefile"
export PERFVM_KERNEL=${PERFVM_KERNEL:-$W/vmlinuz}
for v in plain neon; do
  if [ $v = plain ]; then fpu=vfpv3-d16; else fpu=neon; fi
  make -s -C "$S" CC="$ROOT/scripts/jibo-cc.sh" OUT="$B/$v-build" CFLAGS="-O2 -mfpu=$fpu -DJD_PROFILE" \
    "$B/$v-build/bench-engine"
  printf '/bin/%s\t/work/model.jdw\t/work/tokenizer.jdt\t/work/request1.json\t/work/request3.json\t%s\n' \
    "$v" "$WL" > "$B/jobs-$v"
  TMPDIR="$B/vm-$v" "$HERE/run.sh" "$B/jobs-$v" bin/$v="$B/$v-build/bench-engine" \
    work/model.jdw="$MODEL/model.jdw" work/tokenizer.jdt="$MODEL/tokenizer.jdt" \
    work/request1.json="$HERE/fixtures/request1.json" work/request3.json="$HERE/fixtures/request3.json" \
    > "$B/console-$v.txt" &
done
wait
python3 - "$B" "$OUT/$LABEL.jsonl" "$LABEL" "$REF" <<'PY'
import json, subprocess, sys
b, out, label, ref = sys.argv[1:]
rows = []
for v in ("plain", "neon"):
    for line in open(f"{b}/console-{v}.txt"):
        if line.startswith("{"):
            r = json.loads(line)
            r.update(variant=v, label=label, model="v22 full (24 layers)", ref=ref)
            rows.append(r)
if len(rows) != 2:
    sys.exit(f"expected 2 results, got {len(rows)}: see {b}/console-*.txt")
open(out, "w").write("".join(json.dumps(r) + "\n" for r in rows))
for r in rows:
    print(f"{r['variant']:6} {r['workload']:10} {r['total']/1e9:8.2f} G instructions")
PY
