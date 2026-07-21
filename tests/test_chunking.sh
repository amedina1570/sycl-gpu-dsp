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

echo "--- checking chunked output against an independent ground truth ---"
# The byte-identical check above only proves wide/narrow chunking agree with
# EACH OTHER, not that either is actually correct. This computes a handful
# of frames straight from the raw file in numpy -- decode -> Hann window ->
# FFT -> dB -> fftshift, entirely independent of iq2spectrogram's C++/cuFFT
# implementation -- and diffs them against the many-small-chunks run, so a
# bug in the windowing/FFT/dB/fftshift math itself (applied to whichever
# samples got read) would be caught even if it happened to affect wide and
# narrow chunking identically.
#
# What this does NOT catch: a seek offset that's wrong by a small constant
# for every chunk. Magnitude spectra are shift-invariant (Fourier shift
# theorem), so a uniform few-sample misread is invisible to any check based
# on spectrogram output, no matter how it's computed -- see
# cli::chunk_start_sample in cli_util.hpp and its unit test in
# test_cli_util.cpp, which is where that bug class is actually caught.
python3 -c "
import numpy as np, sys

NFFT, HOP = 8192, 2048
raw = np.fromfile('$WORK/test.sigmf-data', dtype=np.int16).reshape(-1, 2)
iq = (raw[:, 0].astype(np.float64) + 1j * raw[:, 1].astype(np.float64)) / 32768.0
nframes = (len(iq) - NFFT) // HOP + 1

n = np.arange(NFFT)
window = 0.5 * (1.0 - np.cos(2*np.pi*n/(NFFT-1)))  # must match dsp::hann_coeff exactly

spec = np.memmap('$WORK/narrow_spectrogram.bin', dtype=np.float32).reshape(-1, NFFT)

ok = True
for idx in (0, nframes // 2, nframes - 1):
    frame = iq[idx*HOP : idx*HOP + NFFT] * window
    fft = np.fft.fft(frame)
    db = 10*np.log10(np.abs(fft)**2 + 1e-12)
    expected = np.fft.fftshift(db)
    actual = spec[idx].astype(np.float64)
    # Only bins within 30dB of the frame's peak: cuFFT (single precision,
    # different summation order) vs this independent numpy DFT diverge by
    # tens of dB deep in the noise floor from rounding alone, same caveat
    # test_cufft_batch.cpp documents -- not a correctness issue there.
    mask = expected >= (expected.max() - 30)
    err = np.max(np.abs(expected[mask] - actual[mask]))
    print(f'  frame {idx}: max err (near-peak bins) = {err:.3f} dB')
    if err > 1.0:
        ok = False
sys.exit(0 if ok else 1)
"
check $? "chunked output matches an independently-computed ground truth"

echo
echo "pass=$pass fail=$fail"
[[ "$fail" -eq 0 ]]
