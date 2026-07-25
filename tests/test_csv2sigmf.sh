#!/usr/bin/env bash
# Integration smoke test for csv2sigmf: generates a small synthetic vendor-
# style CSV IQ file (header + a known tone, as plain text), converts it with
# csv2sigmf, and runs iq2spectrogram on the result to confirm the recovered
# tone lands where expected -- i.e. csv2sigmf's binary output is byte-for-
# byte usable by the rest of the pipeline with no extra steps. Also checks
# that a malformed row and an out-of-range value both fail loudly instead of
# silently producing corrupt output.
#
# Needs g++ (for csv2sigmf) and the acpp toolchain + GPU (for
# iq2spectrogram); source env/acpp-env.sh first.
#
# Usage: ./tests/test_csv2sigmf.sh
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

echo "--- building csv2sigmf and iq2spectrogram ---"
make -C "$ROOT" "$BUILD/csv2sigmf" "$BUILD/iq2spectrogram"
export LD_LIBRARY_PATH="$(make -C "$ROOT" -s print-cuda-path)/lib64:${LD_LIBRARY_PATH:-}"

echo "--- generating synthetic vendor-style CSV (header + tone at bin 137) ---"
python3 -c "
import numpy as np
rng = np.random.default_rng(0)
nsamp = 200_000
tone = 0.3 * np.cos(2 * np.pi * 137 * np.arange(nsamp) / 8192.0)
noise = rng.normal(0, 0.02, nsamp)
i = np.clip((tone + noise) * 30000, -32767, 32767).astype(np.int64)
q = np.zeros(nsamp, dtype=np.int64)
with open('$WORK/IQ.csv', 'w') as f:
    f.write('I Data,Q Data\n')
    np.savetxt(f, np.column_stack([i, q]), fmt='%d,%d')
"

echo "--- running csv2sigmf ---"
"$BUILD/csv2sigmf" "$WORK/IQ.csv" --fs 20e6 --fc 0 -o "$WORK/IQ"
cat "$WORK/IQ.sigmf-meta"

echo "--- checking round-trip sample count ---"
python3 -c "
import numpy as np, sys
csv_rows = sum(1 for _ in open('$WORK/IQ.csv')) - 1  # minus header
raw = np.fromfile('$WORK/IQ.sigmf-data', dtype=np.int16)
sys.exit(0 if raw.size == csv_rows * 2 else 1)
"
check $? "csv2sigmf wrote exactly one ci16_le sample pair per CSV data row"

echo "--- running iq2spectrogram on the converted file ---"
"$BUILD/iq2spectrogram" "$WORK/IQ.sigmf-data" --no-plot -o "$WORK/spec"

echo "--- checking recovered tone bin ---"
python3 -c "
import numpy as np, sys
NFFT = 8192
spec = np.fromfile('$WORK/spec_spectrogram.bin', dtype=np.float32).reshape(-1, NFFT)
peak = spec[0].argmax()
expected = {NFFT // 2 - 137, NFFT // 2 + 137}
sys.exit(0 if peak in expected else 1)
"
check $? "iq2spectrogram recovers the known tone from csv2sigmf's output (bin $((8192 / 2))±137)"

echo "--- checking a malformed row fails loudly ---"
printf 'I Data,Q Data\n1,2\nNOT,A,ROW\n3,4\n' > "$WORK/bad.csv"
if "$BUILD/csv2sigmf" "$WORK/bad.csv" --fs 1e6 -o "$WORK/bad" 2>"$WORK/bad.err"; then
  check 1 "malformed row is rejected (csv2sigmf unexpectedly succeeded)"
else
  check 0 "malformed row is rejected"
  grep -q "malformed line" "$WORK/bad.err"
  check $? "error message names the offending line"
fi

echo "--- checking an out-of-range ci16 value fails loudly ---"
printf 'I Data,Q Data\n1,2\n999999,4\n' > "$WORK/range.csv"
if "$BUILD/csv2sigmf" "$WORK/range.csv" --fs 1e6 -o "$WORK/range" 2>"$WORK/range.err"; then
  check 1 "out-of-range value is rejected (csv2sigmf unexpectedly succeeded)"
else
  check 0 "out-of-range value is rejected"
fi

echo
echo "pass=$pass fail=$fail"
[[ "$fail" -eq 0 ]]
