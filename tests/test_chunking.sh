#!/usr/bin/env bash
# Integration smoke test for iq2spectrogram's chunked/streaming processing
# (see cli::resolve_chunk_frames in src/cli_util.hpp): generates a synthetic
# IQ file, runs it once with the default chunk size and once forced into
# many small chunks, and checks that the two runs produce byte-identical
# output -- i.e. chunking is purely a memory-bounding strategy, not a
# behavior change. Also exercises the memmap+downsample path in
# view/view_spec.py so it's confirmed not to choke on the result.
#
# Needs the acpp toolchain (`source env/acpp-env.sh` first), a CUDA-backed
# GPU, and python3 with numpy (+ matplotlib for the plotting check).
#
# Usage: ./tests/test_chunking.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

pass=0
fail=0
check() {
  if [[ "$1" -eq 0 ]]; then echo "  OK: $2"; pass=$((pass+1));
  else echo "  FAIL: $2"; fail=$((fail+1)); fi
}

if ! command -v acpp >/dev/null 2>&1; then
  echo "acpp not found on PATH -- run 'source env/acpp-env.sh' first" >&2
  exit 1
fi
if ! python3 -c "import numpy" >/dev/null 2>&1; then
  echo "python3 + numpy required to generate the synthetic test file" >&2
  exit 1
fi

echo "--- building iq2spectrogram ---"
make -C "$ROOT" "$BUILD/iq2spectrogram"
export LD_LIBRARY_PATH="$(make -C "$ROOT" -s print-cuda-path)/lib64:${LD_LIBRARY_PATH:-}"

echo "--- generating synthetic ci16_le IQ file (~340 MB, tone at bin 137) ---"
python3 -c "
import numpy as np
rng = np.random.default_rng(0)
nsamp = 85_000_000
tone = 0.3*np.cos(2*np.pi*137*np.arange(nsamp)/8192.0)
noise = rng.normal(0, 0.02, nsamp)
i = np.clip((tone+noise)*30000, -32767, 32767).astype(np.int16)
q = np.zeros(nsamp, dtype=np.int16)
iq = np.empty(nsamp*2, dtype=np.int16)
iq[0::2] = i; iq[1::2] = q
iq.tofile('$WORK/test.sigmf-data')
"

echo "--- run 1: default chunk-mb (few large chunks) ---"
"$BUILD/iq2spectrogram" "$WORK/test.sigmf-data" --fs 20e6 --fc 0 --no-plot -o "$WORK/wide" \
  | tee "$WORK/wide.log" | grep -E "chunk\(s\) total"

echo "--- run 2: --chunk-mb 4 (many small chunks) ---"
"$BUILD/iq2spectrogram" "$WORK/test.sigmf-data" --fs 20e6 --fc 0 --no-plot --chunk-mb 4 -o "$WORK/narrow" \
  | tee "$WORK/narrow.log" | grep -E "chunk\(s\) total"

wide_chunks=$(grep -oP '\d+(?= chunk\(s\) total)' "$WORK/wide.log")
narrow_chunks=$(grep -oP '\d+(?= chunk\(s\) total)' "$WORK/narrow.log")
echo "wide=$wide_chunks chunk(s), narrow=$narrow_chunks chunk(s)"
check $([[ "$narrow_chunks" -gt "$wide_chunks" ]]; echo $?) "--chunk-mb actually changes the chunk count"

cmp -s "$WORK/wide_spectrogram.bin" "$WORK/narrow_spectrogram.bin"
check $? "chunked and non-chunked runs produce byte-identical .bin output"

echo "--- checking recovered tone bin ---"
python3 -c "
import numpy as np, sys
NFFT = 8192
spec = np.fromfile('$WORK/wide_spectrogram.bin', dtype=np.float32).reshape(-1, NFFT)
peak = spec[0].argmax()
# The tone is a real cosine, so its spectrum has exactly-symmetric peaks at
# +-137; FFT rounding decides which one argmax lands on. Accept either.
expected = {NFFT//2 - 137, NFFT//2 + 137}
sys.exit(0 if peak in expected else 1)
"
check $? "peak bin matches the known input tone (bin $((8192/2))±137)"

if python3 -c "import matplotlib" >/dev/null 2>&1; then
  echo "--- checking view_spec.py handles the (downsampled) output ---"
  python3 "$ROOT/view/view_spec.py" "$WORK/wide_spectrogram.json" "$WORK/wide_spectrogram.png" >/dev/null
  check $([[ -s "$WORK/wide_spectrogram.png" ]]; echo $?) "view_spec.py produced a non-empty PNG"
else
  echo "  (skipping view_spec.py check: matplotlib not installed)"
fi

echo
echo "pass=$pass fail=$fail"
[[ "$fail" -eq 0 ]]
