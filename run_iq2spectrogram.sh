#!/usr/bin/env bash
# Convenience wrapper: builds iq2spectrogram via the Makefile if needed (which
# auto-detects acpp/CUDA/GPU arch), then runs it. All build configuration lives
# in the Makefile; this script only adds the runtime library paths.
# Usage: ./run_iq2spectrogram.sh <iq-file> [iq2spectrogram options...]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$ROOT/build/iq2spectrogram"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <iq-file> [iq2spectrogram options...]" >&2
  exit 1
fi

source "$ROOT/env/acpp-env.sh"

# Build (make is a no-op when the binary is already up to date).
make -C "$ROOT" "$BIN"

# cuFFT/cudart live under the CUDA the Makefile linked against; make sure the
# loader can find them at runtime.
CUDA_PATH="$(make -C "$ROOT" -s print-cuda-path)"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:${LD_LIBRARY_PATH:-}"

exec "$BIN" "$@"
