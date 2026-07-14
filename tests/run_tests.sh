#!/usr/bin/env bash
# Build and run the unit test suite.
#
# Host tests (tests/host/*.cpp) are plain C++, no SYCL/GPU required.
# GPU tests (tests/gpu/*.cpp) exercise real SYCL/cuFFT kernels on the GPU
# and need the acpp toolchain (`source env/acpp-env.sh` first) plus a CUDA
# device. Adjust SM_ARCH / CUDA_PATH below to match your GPU/toolkit, same
# as the raw compile commands in the README.
#
# Usage:
#   ./tests/run_tests.sh          # host + GPU tests
#   ./tests/run_tests.sh --host   # host tests only, no acpp/GPU needed
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src"
TESTS="$ROOT/tests"
BUILD="$ROOT/build/tests"
mkdir -p "$BUILD"

SM_ARCH="${SM_ARCH:-sm_75}"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda-12.6}"

HOST_ONLY=0
[[ "${1:-}" == "--host" ]] && HOST_ONLY=1

pass=0
fail=0

run_bin() {
  local name="$1" bin="$2"
  echo "== $name =="
  if "$bin"; then
    pass=$((pass+1))
  else
    fail=$((fail+1))
  fi
  echo
}

echo "--- host tests (plain g++, no GPU) ---"
g++ -O1 -std=c++17 -I"$SRC" -I"$TESTS" \
  "$TESTS/host/main.cpp" \
  "$TESTS/host/test_sigmf_meta.cpp" \
  "$TESTS/host/test_cli_util.cpp" \
  "$TESTS/host/test_dsp_math.cpp" \
  -o "$BUILD/host_tests"
run_bin "host_tests" "$BUILD/host_tests"

if [[ "$HOST_ONLY" -eq 1 ]]; then
  echo "pass=$pass fail=$fail"
  [[ "$fail" -eq 0 ]]
  exit $?
fi

if ! command -v acpp >/dev/null 2>&1; then
  echo "acpp not found on PATH -- run 'source env/acpp-env.sh' first, or pass --host to skip GPU tests" >&2
  exit 1
fi

echo "--- GPU tests: FFT vs DFT (pure SYCL, generic target) ---"
acpp -O2 --acpp-targets=generic -I"$SRC" -I"$TESTS" \
  "$TESTS/gpu/test_fft_vs_dft.cpp" -o "$BUILD/test_fft_vs_dft"
run_bin "test_fft_vs_dft" "$BUILD/test_fft_vs_dft"

echo "--- GPU tests: Hann window kernel (pure SYCL, generic target) ---"
acpp -O2 --acpp-targets=generic -I"$SRC" -I"$TESTS" \
  "$TESTS/gpu/test_window_kernel.cpp" -o "$BUILD/test_window_kernel"
run_bin "test_window_kernel" "$BUILD/test_window_kernel"

echo "--- GPU tests: batched cuFFT interop (cuda target + cuFFT link) ---"
acpp -O2 --acpp-targets=cuda:$SM_ARCH -I"$SRC" -I"$TESTS" \
  -I"$CUDA_PATH/include" -L"$CUDA_PATH/lib64" -lcufft -lcudart \
  "$TESTS/gpu/test_cufft_batch.cpp" -o "$BUILD/test_cufft_batch"
run_bin "test_cufft_batch" "$BUILD/test_cufft_batch"

echo "pass=$pass fail=$fail"
[[ "$fail" -eq 0 ]]
