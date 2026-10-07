#!/usr/bin/env bash
# robot-run.sh STEP: the robot-side steps of docs/jibo-port.md, for the owner to run.
#
# Every step is a deployment step: run it only within the agreed scope (which robot, which
# directory, which user). Nothing here flashes, overwrites system files, stops services, changes
# clocks or thermal settings, or commands motors. It copies into one directory and `cleanup`
# removes that directory.
#
# Environment (all required; nothing is guessed):
#   JIBO_SSH    the SSH destination, e.g. jibo-skill@jibo.local (prefer the jibo-skill user)
#   JIBO_DIR    an isolated directory on the robot, e.g. /tmp/jibo-decider-test
#   JIBO_ENV    the shell prefix that reproduces the game host's run.sh display access and
#               library shim, e.g. 'DISPLAY=:0 LD_LIBRARY_PATH=/path/to/shim' (from Phase 0)
#   BUILD       the cross-built binaries (scripts/build-jibo.sh output)
#   EXPORT      an export directory (model.jdw, tokenizer.jdt), for bench, ask and serve-test
#
# Steps:
#   deploy       copy the binaries (and EXPORT, if set) to JIBO_DIR
#   probe        run jibo-gl-probe; results/probe-<time>.jsonl
#   bench        bench-op on an MLP projection: a sweep of dispatch sizes and tokens per thread
#   ask          one CPU request (512-token window), with the peak resident set from /proc
#   status       thermal zones, clocks as read-only files report them, MemAvailable, load
#   cleanup      remove JIBO_DIR
set -euo pipefail
STEP=${1:?usage: robot-run.sh deploy|probe|bench|ask|status|cleanup}
: "${JIBO_SSH:?set JIBO_SSH}" "${JIBO_DIR:?set JIBO_DIR}"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RESULTS="$HERE/results"
mkdir -p "$RESULTS"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
remote() { ssh -o BatchMode=yes "$JIBO_SSH" "$@"; }

case "$STEP" in
  deploy)
    : "${BUILD:?set BUILD}"
    remote "mkdir -p '$JIBO_DIR'"
    scp -q "$BUILD/jibo-decider" "$BUILD/jibo-gl-probe" "$BUILD/bench-op" "$JIBO_SSH:$JIBO_DIR/"
    if [ -n "${EXPORT:-}" ]; then scp -q "$EXPORT/model.jdw" "$EXPORT/tokenizer.jdt" "$JIBO_SSH:$JIBO_DIR/"; fi
    remote "cd '$JIBO_DIR' && ls -la && sha256sum *" | tee "$RESULTS/deploy-$STAMP.txt"
    ;;
  probe)
    : "${JIBO_ENV:?set JIBO_ENV}"
    remote "cd '$JIBO_DIR' && $JIBO_ENV ./jibo-gl-probe --reps 50" | tee "$RESULTS/probe-$STAMP.jsonl"
    ;;
  bench)
    : "${JIBO_ENV:?set JIBO_ENV}"
    out="$RESULTS/bench-$STAMP.jsonl"
    for tokens in 32 128; do
      for tt in 4 8 16; do
        for rows in 64 256 1024 4096; do
          remote "cd '$JIBO_DIR' && $JIBO_ENV ./bench-op model.jdw l0.up --tokens $tokens --tt $tt \
                  --rows-per-dispatch $rows --reps 5" | tee -a "$out"
        done
      done
    done
    ;;
  ask)
    req='{"state":"Jibo, can you set a timer for ten minutes?","questions":{"addressed":{"type":"noul","instructions":"Is the speaker talking to the robot?"},"intent":{"type":"choice","instructions":"What does the speaker want?","criteria":{"timer":"set a timer or alarm","weather":"weather","music":"play music","chat":"small talk"}}}}'
    remote "cd '$JIBO_DIR' && echo '$req' | ./jibo-decider ask model.jdw tokenizer.jdt --window 512 & pid=\$!; \
            peak=0; while kill -0 \$pid 2>/dev/null; do r=\$(awk '/VmHWM/{print \$2}' /proc/\$pid/status 2>/dev/null); \
            [ -n \"\$r\" ] && [ \"\$r\" -gt \"\$peak\" ] && peak=\$r; sleep 0.2; done; wait \$pid; echo \"{\\\"vm_hwm_kb\\\":\$peak}\"" \
      | tee "$RESULTS/ask-$STAMP.jsonl"
    ;;
  status)
    remote 'for z in /sys/class/thermal/thermal_zone*; do echo "$(cat $z/type 2>/dev/null) $(cat $z/temp 2>/dev/null)"; done;
            grep -E "MemAvailable|MemTotal" /proc/meminfo; cat /proc/loadavg' | tee "$RESULTS/status-$STAMP.txt"
    ;;
  cleanup)
    remote "rm -rf '$JIBO_DIR'" && echo "removed $JIBO_DIR on $JIBO_SSH"
    ;;
  *) echo "unknown step $STEP" >&2; exit 2 ;;
esac
