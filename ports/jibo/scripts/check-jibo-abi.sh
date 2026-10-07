#!/usr/bin/env bash
# check-jibo-abi.sh BINARY...: will these ELF files load on Jibo?
#
# Jibo runs 32-bit ARMv7, hard float, glibc 2.21, libstdc++ up to GLIBCXX_3.4.20, and has no
# development files. For each binary this checks: ELF32 ARM; the hard-float ABI attribute; the
# armhf loader; NEEDED libraries limited to libc, libm, libdl, libpthread and the loader (the
# runtime is C and opens libGL itself); no symbol version above GLIBC_2.21; and no GLIBCXX or
# CXXABI version at all. It reads files only and runs nothing.
#
# If the owner's existing checker is available, set JIBO_ABI_CHECKER to its path and it runs
# too, after these checks. READELF defaults to arm-linux-gnueabihf-readelf.
set -euo pipefail
READELF=${READELF:-arm-linux-gnueabihf-readelf}
MAX_GLIBC=2.21
[ $# -gt 0 ] || { echo "usage: check-jibo-abi.sh BINARY..." >&2; exit 2; }
command -v "$READELF" >/dev/null || { echo "no $READELF (set READELF)" >&2; exit 2; }

ver_gt() { [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | tail -1)" = "$1" ] && [ "$1" != "$2" ]; }

status=0
for bin in "$@"; do
  problems=()
  hdr=$("$READELF" -h "$bin")
  grep -q "Class:.*ELF32" <<<"$hdr" || problems+=("not ELF32")
  grep -q "Machine:.*ARM" <<<"$hdr" || problems+=("not ARM")
  grep -q "hard-float ABI" <<<"$hdr" || problems+=("not the hard-float ABI")
  "$READELF" -l "$bin" | grep -q "/lib/ld-linux-armhf.so.3" || problems+=("loader is not /lib/ld-linux-armhf.so.3")
  while read -r lib; do
    case "$lib" in
      libc.so.6|libm.so.6|libdl.so.2|libpthread.so.0|ld-linux-armhf.so.3) ;;
      *) problems+=("needs $lib") ;;
    esac
  done < <("$READELF" -d "$bin" | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p')
  vers=$("$READELF" -V "$bin" | grep -o "Name: [A-Z_]*[0-9.]*" | awk '{print $2}' | sort -u)
  for v in $vers; do
    case "$v" in
      GLIBC_[0-9]*) ver_gt "${v#GLIBC_}" "$MAX_GLIBC" && problems+=("needs $v (Jibo has $MAX_GLIBC)") ;;
      GLIBCXX_*|CXXABI_*) problems+=("needs $v (the runtime must not link libstdc++)") ;;
    esac
  done
  if [ ${#problems[@]} -eq 0 ]; then
    echo "ok   $bin  ($(tr '\n' ' ' <<<"$vers"))"
  else
    status=1
    for p in "${problems[@]}"; do echo "FAIL $bin: $p"; done
  fi
done
if [ -n "${JIBO_ABI_CHECKER:-}" ]; then
  "$JIBO_ABI_CHECKER" "$@" || status=1
fi
exit $status
