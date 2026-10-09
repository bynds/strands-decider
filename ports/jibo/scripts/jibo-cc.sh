#!/usr/bin/env bash
# jibo-cc.sh: a C compiler driver for ARMv7 hard-float against a glibc <= 2.21 sysroot
# (fetch-sysroot.sh), from this host's arm-linux-gnueabihf-gcc. The caller picks the FPU
# (-mfpu=vfpv3-d16 for the shipped build, -mfpu=neon for the NEON one); the default is vfpv3-d16.
#
#   JIBO_SYSROOT   the sysroot directory (required)
#
# The host cross gcc searches its own (newer glibc) directories before any --sysroot, so the
# sysroot's headers, start files and libraries are named explicitly, as in build-jibo.sh.
set -euo pipefail
: "${JIBO_SYSROOT:?set JIBO_SYSROOT (scripts/fetch-sysroot.sh output)}"
R=$(cd "$JIBO_SYSROOT" && pwd)
M=arm-linux-gnueabihf
CC=${JIBO_HOST_CROSS_CC:-arm-linux-gnueabihf-gcc}
GCCINC=$("$CC" -print-file-name=include)
link=1
for a in "$@"; do case "$a" in -c|-S|-E) link=0 ;; esac; done
ARGS=(-march=armv7-a -mfpu=vfpv3-d16 -mfloat-abi=hard
      -nostdinc -isystem "$GCCINC" -isystem "$R/usr/include/$M" -isystem "$R/usr/include")
if [ $link = 1 ]; then
  ARGS+=(-B"$R/usr/lib/$M/" -L"$R/usr/lib/$M" -L"$R/lib/$M"
         -Wl,--sysroot="$R" -Wl,-rpath-link,"$R/lib/$M" -Wl,-rpath-link,"$R/usr/lib/$M"
         -Wl,--dynamic-linker=/lib/ld-linux-armhf.so.3 -Wl,--hash-style=both -static-libgcc)
fi
exec "$CC" "${ARGS[@]}" "$@"
