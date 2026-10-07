#!/usr/bin/env bash
# evaluate-export.sh CHECKPOINT EXPORT_DIR OUT_DIR: score a .jdw export against its checkpoint.
#
# 1. parity: tests/engine_parity.py, the export's answers against the Python engine's (fp32, CPU)
#    on evaluation/device_parity.py's 54 answers -> OUT_DIR/parity.json, parity.log
# 2. JevBench: evaluation/jevbench/jevbench.sh with the C runtime standing in for the server
#    (tools/jevbench_serve.py), at the checkpoint's window and, with WINDOWS="4096 512", at each
#    -> OUT_DIR/jevbench_w<window>/
#
# Environment: JIBO_BIN (default build/jibo-decider), PY (default python3), WINDOWS (default:
# the checkpoint's max_length), SKIP_PARITY=1, SKIP_JEVBENCH=1, JEVBENCH_DIR (where to clone it).
# CHECKPOINT must be a local directory: jevbench.sh checks /health against its path.
set -euo pipefail
CKPT=${1:?usage: evaluate-export.sh CHECKPOINT EXPORT_DIR OUT_DIR}
EXPORT=$(cd "${2:?}" && pwd)
OUT=${3:?}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPO=$(cd "$HERE/../.." && pwd)
PY=${PY:-$(command -v python3)}
BIN=$(cd "$(dirname "${JIBO_BIN:-$HERE/build/jibo-decider}")" && pwd)/$(basename "${JIBO_BIN:-jibo-decider}")
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)

if [ -z "${SKIP_PARITY:-}" ]; then
  "$PY" "$HERE/tests/engine_parity.py" "$BIN" "$EXPORT/model.jdw" "$EXPORT/tokenizer.jdt" "$CKPT" \
    --tol 1 --out "$OUT/parity.json" | tee "$OUT/parity.log"
fi

if [ -z "${SKIP_JEVBENCH:-}" ]; then
  CFG="$CKPT/strands_decider_config.json"
  [ -f "$CFG" ] || CFG="$CKPT/hobson_config.json"
  for w in ${WINDOWS:-$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["max_length"])' "$CFG")}; do
    # jevbench.sh checks that /health reports the window the checkpoint's config names, so each
    # window gets a copy of the checkpoint whose config says it.
    ck="$OUT/ckpt_w$w"
    rm -rf "$ck"
    cp -rL "$CKPT" "$ck"
    "$PY" - "$ck" "$w" <<'EOF'
import json, os, sys
d, w = sys.argv[1], int(sys.argv[2])
p = os.path.join(d, "strands_decider_config.json")
if not os.path.exists(p):
    p = os.path.join(d, "hobson_config.json")
c = json.load(open(p)); c["max_length"] = w; json.dump(c, open(p, "w"), indent=2)
EOF
    HOBSON="$HERE/tools/jevbench_serve.py" JIBO_EXPORT="$EXPORT" JIBO_BIN="$BIN" PY="$PY" GPU=0 \
      PORT="${PORT:-8199}" JEVBENCH_DIR="${JEVBENCH_DIR:-$OUT/jevbench-src}" \
      bash "$REPO/evaluation/jevbench/jevbench.sh" "$ck" "$OUT/jevbench_w$w"
    rm -rf "$ck"
  done
fi
