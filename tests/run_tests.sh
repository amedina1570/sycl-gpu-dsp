#!/usr/bin/env bash
# Build and run the unit test suite. Compilation is delegated to the Makefile
# (which auto-detects acpp / CUDA / GPU arch); this script just builds the test
# binaries and tallies pass/fail. Override detection the same way as make, e.g.
# CUDA_PATH=... SM_ARCH=... ./tests/run_tests.sh.
#
# Host tests (tests/host/*.cpp) are plain C++, no SYCL/GPU required.
# GPU tests (tests/gpu/*.cpp) exercise real SYCL/cuFFT kernels and need the
# acpp toolchain (`source env/acpp-env.sh` first) plus a CUDA device.
#
# Usage:
#   ./tests/run_tests.sh          # host + GPU tests
#   ./tests/run_tests.sh --host   # host tests only, no acpp/GPU needed
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_BUILD="$ROOT/build/tests"

HOST_ONLY=0
[[ "${1:-}" == "--host" ]] && HOST_ONLY=1

pass=0
fail=0
run_bin() {
  local name="$1" bin="$2"
  echo "== $name =="
  if "$bin"; then pass=$((pass+1)); else fail=$((fail+1)); fi
  echo
}

echo "--- host tests (plain g++, no GPU) ---"
make -C "$ROOT" host-tests
run_bin "host_tests" "$TEST_BUILD/host_tests"

if [[ "$HOST_ONLY" -eq 1 ]]; then
  echo "pass=$pass fail=$fail"
  [[ "$fail" -eq 0 ]]
  exit $?
fi

if ! command -v acpp >/dev/null 2>&1; then
  echo "acpp not found on PATH -- run 'source env/acpp-env.sh' first, or pass --host to skip GPU tests" >&2
  exit 1
fi

echo "--- GPU tests (SYCL FFT/DFT/window + cuFFT interop) ---"
make -C "$ROOT" gpu-tests

# cuFFT/cudart live under the CUDA the Makefile linked against.
export LD_LIBRARY_PATH="$(make -C "$ROOT" -s print-cuda-path)/lib64:${LD_LIBRARY_PATH:-}"

run_bin "test_fft_vs_dft"    "$TEST_BUILD/test_fft_vs_dft"
run_bin "test_window_kernel" "$TEST_BUILD/test_window_kernel"
run_bin "test_cufft_batch"   "$TEST_BUILD/test_cufft_batch"

echo "pass=$pass fail=$fail"
[[ "$fail" -eq 0 ]]
