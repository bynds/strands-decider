#!/usr/bin/env bash
# run.sh JOBS [DEST=SRC ...]: run ARMv7 programs in a full-system qemu with exact instruction
# counting, and print what they print.
#
# The guest is an arm64 kernel (fetch-kernel.sh) running the armhf programs in AArch32 EL0, with
# the glibc 2.21 stand-in sysroot's loader and libraries. qemu runs with -icount shift=0, so the
# emulated PMU's instructions-retired event is exact and repeatable: programs read it through
# perf_event_open (perfvm/jd_icount.h). Only instruction
# counts mean anything here; qemu's time does not model a Cortex-A15.
#
# JOBS: one command per line, arguments separated by tabs, paths inside the guest. Each DEST=SRC
# puts a file into the guest (programs under bin/, data under work/, the jobs' working directory).
#   JIBO_SYSROOT   the stand-in sysroot (scripts/fetch-sysroot.sh)
#   PERFVM_KERNEL  the guest kernel (default: fetch-kernel.sh into $TMPDIR/perfvm)
#   PERFVM_MEM     guest memory in MB (default 3072: the initramfs holds the model)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
JOBS=${1:?usage: run.sh JOBS [DEST=SRC ...]}
shift
: "${JIBO_SYSROOT:?set JIBO_SYSROOT}"
W=${TMPDIR:-/tmp}/perfvm
mkdir -p "$W"
KERNEL=${PERFVM_KERNEL:-$W/vmlinuz}
[ -f "$KERNEL" ] || "$HERE/fetch-kernel.sh" "$W" >/dev/null
[ -f "$W/init" ] && [ "$W/init" -nt "$HERE/init.c" ] ||
  arm-linux-gnueabihf-gcc -static -O2 -Wall -o "$W/init" "$HERE/init.c"
L=$JIBO_SYSROOT/lib/arm-linux-gnueabihf
libs=()
for f in ld-linux-armhf.so.3 libc.so.6 libm.so.6 libpthread.so.0 libdl.so.2 librt.so.1 libgcc_s.so.1; do
  libs+=("lib/arm-linux-gnueabihf/$f=$L/$f")
done
python3 "$HERE/mkinitramfs.py" "$W/initramfs.cpio" init="$W/init" jobs="$JOBS" work/ \
  lib/ld-linux-armhf.so.3="$L/ld-linux-armhf.so.3" "${libs[@]}" "$@"
qemu-system-aarch64 -M virt -cpu cortex-a57 -m "${PERFVM_MEM:-3072}" -nographic -no-reboot \
  -icount shift=0 -kernel "$KERNEL" -initrd "$W/initramfs.cpio" \
  -append "console=ttyAMA0 rdinit=/init quiet loglevel=1" </dev/null |
  tr -d '\r' | grep -v -e '^\[' -e '^EFI' || true
