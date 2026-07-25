#!/usr/bin/env bash
# Integration smoke test for radar_pulses (the GPU port of
# view_radar_pulses.py's pulse-train detection): synthesizes a
# rectangular-envelope pulse train with known PRI/width as a complex-
# baseband ci16_le IQ file, runs radar_pulses on it, checks the reported
# stats against the known ground truth, cross-checks against the same file
# run through the all-NumPy view_radar_pulses.py path (two independent
# implementations should agree), and confirms view_radar_pulses.py can plot
# radar_pulses' precomputed .json/.bin output.
#
# Needs the acpp toolchain (`source env/acpp-env.sh` first), a GPU, and
# python3 with numpy (+ matplotlib for the plotting check).
#
# Usage: ./tests/test_radar_pulses.sh
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

echo "--- building radar_pulses ---"
make -C "$ROOT" "$BUILD/radar_pulses"

# Ground truth: fs=1MHz, PRI=100us (100 samples), width=10us (10 samples),
# duty=10%, over 5ms (50 periods). The very first pulse starts at sample 0
# with no captured rising edge, so detect_pulses drops it as boundary-cut --
# 49 complete pulses survive. A complex-baseband tone (not a real-valued
# one) is essential here: the envelope of a real cosine oscillates with the
# carrier and would be mis-detected as many short sub-pulses instead of one
# clean pulse per gate.
FS=1000000.0
PRI=0.0001
WIDTH=0.00001
EXPECTED_N=49
echo "--- generating synthetic complex-baseband pulse train (fs=1MHz, PRI=100us, width=10us) ---"
python3 -c "
import numpy as np
fs, pri, width, duration = $FS, $PRI, $WIDTH, 5e-3
n = int(duration * fs)
t = np.arange(n) / fs
gate = (t % pri) < width
carrier = np.exp(1j * 2 * np.pi * 50000 * t)
rng = np.random.default_rng(0)
noise = rng.normal(0, 0.01, n) + 1j * rng.normal(0, 0.01, n)
iq = 0.5 * carrier * gate + noise
i = np.clip(iq.real * 30000, -32767, 32767).astype(np.int16)
q = np.clip(iq.imag * 30000, -32767, 32767).astype(np.int16)
raw = np.empty(n * 2, dtype=np.int16)
raw[0::2] = i; raw[1::2] = q
raw.tofile('$WORK/pulse.sigmf-data')
"

echo "--- running radar_pulses (GPU) ---"
"$BUILD/radar_pulses" "$WORK/pulse.sigmf-data" --fs "$FS" --fc 0 --duration 5e-3 -o "$WORK/gpu"

echo "--- running view_radar_pulses.py (all-NumPy) on the same file ---"
python3 "$ROOT/view/view_radar_pulses.py" "$WORK/pulse.sigmf-data" --fs "$FS" --fc 0 \
  --duration 5e-3 -o "$WORK/legacy.png" | tee "$WORK/legacy.log"

echo "--- checking radar_pulses' stats against known ground truth ---"
python3 -c "
import json, sys
with open('$WORK/gpu_radar.json') as f:
    s = json.load(f)

ok = True
def check(name, got, expected, tol):
    global ok
    err = abs(got - expected)
    status = 'OK' if err <= tol else 'FAIL'
    print(f'  {status}: {name} = {got:.6g} (expected {expected:.6g}, tol {tol:.2g})')
    if err > tol:
        ok = False

check('n_pulses', s['n_pulses'], $EXPECTED_N, 2)
check('pulse_width_mean_s', s['pulse_width_mean_s'], $WIDTH, 2e-6)
check('pri_mean_s', s['pri_mean_s'], $PRI, 2e-6)
check('prf_hz', s['prf_hz'], 1.0 / $PRI, 200.0)
check('duty_cycle', s['duty_cycle'], $WIDTH / $PRI, 0.03)
sys.exit(0 if ok else 1)
"
check $? "radar_pulses' reported stats match the known ground truth"

echo "--- cross-checking radar_pulses against the independent NumPy implementation ---"
python3 -c "
import json, sys
with open('$WORK/gpu_radar.json') as f:
    gpu = json.load(f)
with open('$WORK/legacy.json') as f:
    py = json.load(f)

ok = True
for key in ('n_pulses', 'pulse_width_mean_s', 'pri_mean_s', 'prf_hz', 'duty_cycle'):
    g, p = gpu[key], py[key]
    tol = max(abs(p) * 0.01, 1e-6)  # both run the same algorithm -- float32 vs
                                     # float64 rounding is the only expected gap
    match = abs(g - p) <= tol
    print(f'  {\"OK\" if match else \"FAIL\"}: {key}: gpu={g!r} python={p!r}')
    if not match:
        ok = False
sys.exit(0 if ok else 1)
"
check $? "radar_pulses (SYCL/float32) matches view_radar_pulses.py (NumPy/float64) within 1%"

if python3 -c "import matplotlib" >/dev/null 2>&1; then
  echo "--- checking view_radar_pulses.py can plot radar_pulses' precomputed output ---"
  python3 "$ROOT/view/view_radar_pulses.py" "$WORK/gpu_radar.json" -o "$WORK/precomputed.png" >/dev/null
  check $([[ -s "$WORK/precomputed.png" ]]; echo $?) "view_radar_pulses.py --json produced a non-empty PNG"
else
  echo "  (skipping plot check: matplotlib not installed)"
fi

echo
echo "pass=$pass fail=$fail"
[[ "$fail" -eq 0 ]]
