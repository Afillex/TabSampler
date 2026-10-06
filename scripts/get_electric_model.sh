#!/usr/bin/env bash
# Download the electric string classifier (ADRs 0063, 0064) and check it against its pinned hash.
# Its weights are CC BY 4.0, trained on two CC BY 4.0 datasets: Guitar-TECHS (H. Pedroza,
# W. Abreu, R. M. Corey, I. R. Roman; arXiv:2501.03720) and EGFxSet (H. Pedroza, G. Meza,
# I. R. Roman; doi:10.5281/zenodo.7044411). The code is MIT.
set -euo pipefail
URL="https://github.com/Afillex/TabSampler/releases/download/electric-model-v1/tabsampler-electric-strings-v1.pt"
SHA256="0d46fc2986969a0cd342d54a90cf48091f3878845d9779a087cf37f6bb69f759"
DEST="cache/acoustic/electric/best.pt"

mkdir -p "$(dirname "$DEST")"
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT
curl -fsSL "$URL" -o "$tmp"
got="$(shasum -a 256 "$tmp" | cut -d' ' -f1)"
if [ "$got" != "$SHA256" ]; then
  echo "refusing the download: SHA-256 $got, expected $SHA256" >&2
  exit 1
fi
mv "$tmp" "$DEST"
trap - EXIT
echo "electric string classifier at $DEST"
