#!/usr/bin/env bash
# build-jibo.sh OUT_DIR: cross-build the runtime and the GL probe for Jibo (ARMv7, hard float).
#
# Two ways, by environment:
#
#   JIBO_CC=<the owner's jibo-armcc wrapper>   the intended route: the existing image's Linaro
#                                             GCC 4.8.4 with its target libraries (Phase 0 of
#                                             docs/jibo-port.md recovers the wrapper's path)
#   JIBO_SYSROOT=<dir>                        a stand-in: this host's arm-linux-gnueabihf-gcc
#                                             against a glibc <= 2.21 sysroot (headers, crt
#                                             files and libraries), e.g. Debian 8's glibc 2.19
#                                             packages extracted with dpkg-deb -x
#
# Either way the result is checked by check-jibo-abi.sh, and the build fails if it would not load
# on the robot. Nothing here runs on, or copies to, the robot.
set -euo pipefail
OUT=${1:?usage: build-jibo.sh OUT_DIR}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ARCH="-march=armv7-a -mfpu=neon -mfloat-abi=hard"
mkdir -p "$OUT"

if [ -n "${JIBO_CC:-}" ]; then
  CC="$JIBO_CC"
  CFLAGS="-O2 $ARCH"
  LDFLAGS=""
elif [ -n "${JIBO_SYSROOT:-}" ]; then
  R=$(cd "$JIBO_SYSROOT" && pwd)
  CC=arm-linux-gnueabihf-gcc
  GCCINC=$("$CC" -print-file-name=include)
  M=arm-linux-gnueabihf
  # The host's cross gcc searches its own library directories before any --sysroot, so name
  # the sysroot's headers, start files and libraries explicitly.
  CFLAGS="-O2 $ARCH -nostdinc -isystem $GCCINC -isystem $R/usr/include/$M -isystem $R/usr/include"
  LDFLAGS="-B$R/usr/lib/$M/ -L$R/usr/lib/$M -L$R/lib/$M -Wl,--sysroot=$R -Wl,-rpath-link,$R/lib/$M -static-libgcc"
else
  echo "set JIBO_CC (the jibo-armcc wrapper) or JIBO_SYSROOT (a glibc <= 2.21 armhf sysroot)" >&2
  exit 2
fi

make -s -C "$HERE" OUT="$OUT" CC="$CC" CFLAGS="$CFLAGS" LDFLAGS="$LDFLAGS" \
  "$OUT/jibo-decider" "$OUT/jibo-gl-probe" "$OUT/bench-op"
"$HERE/scripts/check-jibo-abi.sh" "$OUT/jibo-decider" "$OUT/jibo-gl-probe" "$OUT/bench-op"
