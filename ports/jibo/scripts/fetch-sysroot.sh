#!/usr/bin/env bash
# fetch-sysroot.sh DIR: a stand-in Jibo sysroot (armhf, glibc 2.21) for linking on a dev host.
#
# Jibo runs glibc 2.21. Ubuntu 15.04 ("vivid") shipped exactly glibc 2.21 for armhf, so its
# libc6/libc6-dev packages give the same symbol-version ceiling as the robot. This is a
# stand-in for the owner's jibo-armcc image (Linaro GCC 4.8.4 + target libraries); when that
# wrapper is available, build-jibo.sh prefers it (JIBO_CC) and this sysroot is not needed.
#
# Earlier stand-in builds here used Debian 8's glibc 2.19; archive.debian.org is not reachable
# from every build host, which is why this one uses old-releases.ubuntu.com. Every package is
# pinned by SHA-256 and the script refuses a mismatch. Nothing is installed system-wide:
# packages are unpacked with dpkg-deb -x into DIR only.
set -euo pipefail
DIR=${1:?usage: fetch-sysroot.sh DIR}
BASE=https://old-releases.ubuntu.com/ubuntu/pool/main
PKGS=(
  "g/glibc/libc6_2.21-0ubuntu4.3_armhf.deb 7a6aaf93c7ad15d96e303eb916112af66ee2bc981d90e4506d7c9b50d7592865"
  "g/glibc/libc6-dev_2.21-0ubuntu4.3_armhf.deb a7a291e6d374da47eb800060c0037104bc05bd13e6c66562ae7bc0eb716a1899"
  "l/linux/linux-libc-dev_3.19.0-15.15_armhf.deb 94479642604bdd96410e763df71715b1dad96cf4ac1120ad31755537ae5813be"
  "g/gcc-4.9/libgcc1_4.9.1-16ubuntu6_armhf.deb 38d2a1f81b8b0a5b3b475c8a60cc41da3b3780c9cac31ea0e8009d8c76be9cbc"
)
mkdir -p "$DIR/.debs"
R=$(cd "$DIR" && pwd)
for entry in "${PKGS[@]}"; do
  read -r path sum <<<"$entry"
  deb="$R/.debs/$(basename "$path")"
  [ -f "$deb" ] || curl -sSfL --retry 3 -o "$deb" "$BASE/$path"
  echo "$sum  $deb" | sha256sum -c --quiet - || { echo "checksum mismatch: $deb" >&2; exit 1; }
  dpkg-deb -x "$deb" "$R"
done

# Point absolute symlinks (e.g. libm.so -> /lib/arm-linux-gnueabihf/libm.so.6) inside the sysroot.
find "$R" -type l | while read -r l; do
  t=$(readlink "$l")
  case "$t" in /*) ln -sfn "$R$t" "$l" ;; esac
done
# The linker script libc.so names absolute paths; the linker resolves them under --sysroot.
# libgcc1 ships only libgcc_s.so.1; the link needs the unversioned name.
ln -sfn libgcc_s.so.1 "$R/lib/arm-linux-gnueabihf/libgcc_s.so"

printf '%s\n' "${PKGS[@]}" > "$R/PACKAGES.sha256"
echo "sysroot ready: $R (glibc $(ls "$R"/lib/arm-linux-gnueabihf/libc-*.so | sed 's/.*libc-\(.*\)\.so/\1/'))"
