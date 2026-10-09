#!/usr/bin/env bash
# bench.sh LABEL: build the profiling bench (bench_engine.c, -DJD_PROFILE) for ARMv7 from this
# tree, plain (VFPv3-D16, as shipped) and NEON, run both in perfvm, and write
# $PERFVM_OUT/LABEL.jsonl: one line per variant and workload (prefill, request3, prefix_hit) with
# exact user-space instruction counts, in total and per operation.
#
#   JIBO_SYSROOT   the stand-in sysroot (scripts/fetch-sysroot.sh)
#   BENCH_MODEL    directory with model.jdw and tokenizer.jdt (default: $TMPDIR/perfvm/bench4, the
#                  first four layers of the v22 export, tools/truncate_jdw.py)
#   PERFVM_OUT     default: perfvm/results in this tree
set -euo pipefail
LABEL=${1:?usage: bench.sh LABEL}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
: "${JIBO_SYSROOT:?set JIBO_SYSROOT}"
W=${TMPDIR:-/tmp}/perfvm
MODEL=${BENCH_MODEL:-$W/bench4}
OUT=${PERFVM_OUT:-$HERE/results}
B=$W/bin-$LABEL
mkdir -p "$OUT" "$B"
for v in plain neon; do
  if [ $v = plain ]; then fpu=vfpv3-d16; else fpu=neon; fi
  rm -rf "$B/$v-build"  # a fresh build; not make -B, which would regenerate sources in the tree
  make -s -C "$ROOT" CC="$ROOT/scripts/jibo-cc.sh" OUT="$B/$v-build" \
    CFLAGS="-O2 -mfpu=$fpu -DJD_PROFILE" "$B/$v-build/bench-engine"
  cp "$B/$v-build/bench-engine" "$B/$v"
done
J=$B/jobs
: > "$J"
for v in plain neon; do
  printf '/bin/%s\t/work/model.jdw\t/work/tokenizer.jdt\t/work/request1.json\t/work/request3.json\n' "$v" >> "$J"
done
"$HERE/run.sh" "$J" bin/plain="$B/plain" bin/neon="$B/neon" work/model.jdw="$MODEL/model.jdw" \
  work/tokenizer.jdt="$MODEL/tokenizer.jdt" work/request1.json="$HERE/fixtures/request1.json" \
  work/request3.json="$HERE/fixtures/request3.json" > "$B/console.txt"
python3 - "$B/console.txt" "$OUT/$LABEL.jsonl" "$LABEL" <<'PY'
import json, sys
variant, rows = None, []
for line in open(sys.argv[1]):
    if line.startswith('perfvm: job /bin/'):
        variant = line.split('/bin/')[1].split()[0]
    elif line.startswith('{'):
        v = json.loads(line)
        v['variant'], v['label'] = variant, sys.argv[3]
        rows.append(v)
if len(rows) != 6 or any(r['clock'] != 'instructions' for r in rows):
    sys.exit(f'expected 6 instruction-counted results, got {len(rows)}: see {sys.argv[1]}')
open(sys.argv[2], 'w').write(''.join(json.dumps(r) + '\n' for r in rows))
print(f'{len(rows)} results -> {sys.argv[2]}')
for r in rows:
    print(f"{r['variant']:6} {r['workload']:10} {r['total']/1e9:8.3f} G instructions")
PY
