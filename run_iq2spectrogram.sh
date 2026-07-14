#!/usr/bin/env bash
# Convenience wrapper: builds iq2spectrogram if needed, then runs it.
# Usage: ./run_iq2spectrogram.sh <iq-file> [iq2spectrogram options...]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$ROOT/build/iq2spectrogram"
SRC="$ROOT/src/iq2spectrogram.cpp"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <iq-file> [iq2spectrogram options...]" >&2
  exit 1
fi

source "$ROOT/env/acpp-env.sh"

if [[ ! -x "$BIN" || "$SRC" -nt "$BIN" ]]; then
  echo "building iq2spectrogram..."
  acpp -O2 --acpp-targets=cuda:sm_75 "$SRC" -o "$BIN" \
    -I/usr/local/cuda-12.6/include -L/usr/local/cuda-12.6/lib64 -lcufft -lcudart
fi

exec "$BIN" "$@"
