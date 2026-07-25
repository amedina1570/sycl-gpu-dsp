// GPU-executed test for dsp::gpu_smoothed_envelope (radar_lib.hpp), the
// exact envelope + boxcar-smoothing kernels radar_pulses.cpp runs. Checks
// the trivial (unsmoothed) case exactly, and cross-checks smoothing against
// an independently-written host reference implementing the same
// numpy-convolve-'same'-mode formula (see radar_lib.hpp's comment).
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "radar_lib.hpp"
#include <cmath>
#include <cstdint>
#include <sycl/sycl.hpp>

namespace {

std::vector<float> host_envelope_i16(const std::vector<int16_t>& raw) {
  size_t nsamp = raw.size() / 2;
  std::vector<float> env(nsamp);
  for (size_t n = 0; n < nsamp; ++n) {
    float i = dsp::decode_i16(raw[2 * n]);
    float q = dsp::decode_i16(raw[2 * n + 1]);
    env[n] = std::sqrt(i * i + q * q);
  }
  return env;
}

std::vector<float> host_boxcar_same(const std::vector<float>& env, int w) {
  long nsamp = (long)env.size();
  long off = (w - 1) / 2;
  std::vector<float> out(env.size());
  for (long n = 0; n < nsamp; ++n) {
    float acc = 0.0f;
    for (int d = 0; d < w; ++d) {
      long m = n + off - w + 1 + d;
      if (m >= 0 && m < nsamp) acc += env[m];
    }
    out[n] = acc / (float)w;
  }
  return out;
}

std::vector<int16_t> synthetic_iq(size_t nsamp) {
  std::vector<int16_t> raw(nsamp * 2);
  for (size_t n = 0; n < nsamp; ++n) {
    raw[2 * n]     = (int16_t)((n * 37) % 20000 - 10000);
    raw[2 * n + 1] = (int16_t)((n * 13) % 15000 - 7000);
  }
  return raw;
}

}  // namespace

TEST_CASE("gpu_smoothed_envelope with smooth_samples=1 matches per-sample magnitude exactly") {
  constexpr size_t N = 256;
  sycl::queue q{sycl::property::queue::in_order{}};
  std::vector<int16_t> raw = synthetic_iq(N);

  std::vector<float> gpu = dsp::gpu_smoothed_envelope(q, raw, 1);
  std::vector<float> expected = host_envelope_i16(raw);

  REQUIRE(gpu.size() == N);
  for (size_t n = 0; n < N; ++n)
    CHECK(gpu[n] == doctest::Approx(expected[n]).epsilon(0.0005));
}

TEST_CASE("gpu_smoothed_envelope smoothing matches an independent host reference (odd window)") {
  constexpr size_t N = 256;
  constexpr int W = 5;
  sycl::queue q{sycl::property::queue::in_order{}};
  std::vector<int16_t> raw = synthetic_iq(N);

  std::vector<float> gpu = dsp::gpu_smoothed_envelope(q, raw, W);
  std::vector<float> expected = host_boxcar_same(host_envelope_i16(raw), W);

  REQUIRE(gpu.size() == N);
  for (size_t n = 0; n < N; ++n)
    CHECK(gpu[n] == doctest::Approx(expected[n]).epsilon(0.0005));
}

TEST_CASE("gpu_smoothed_envelope smoothing matches an independent host reference (even window)") {
  constexpr size_t N = 128;
  constexpr int W = 8;  // exercises the asymmetric-window branch of the 'same' formula
  sycl::queue q{sycl::property::queue::in_order{}};
  std::vector<int16_t> raw = synthetic_iq(N);

  std::vector<float> gpu = dsp::gpu_smoothed_envelope(q, raw, W);
  std::vector<float> expected = host_boxcar_same(host_envelope_i16(raw), W);

  REQUIRE(gpu.size() == N);
  for (size_t n = 0; n < N; ++n)
    CHECK(gpu[n] == doctest::Approx(expected[n]).epsilon(0.0005));
}

TEST_CASE("gpu_smoothed_envelope also works with the float32 (cf32_le) Raw type") {
  constexpr size_t N = 64;
  sycl::queue q{sycl::property::queue::in_order{}};
  std::vector<float> raw(N * 2);
  for (size_t n = 0; n < N; ++n) {
    raw[2 * n]     = std::sin(0.1f * (float)n);
    raw[2 * n + 1] = std::cos(0.1f * (float)n);
  }

  std::vector<float> gpu = dsp::gpu_smoothed_envelope(q, raw, 1);
  REQUIRE(gpu.size() == N);
  for (size_t n = 0; n < N; ++n) {
    float expected = std::hypot(raw[2 * n], raw[2 * n + 1]);
    CHECK(gpu[n] == doctest::Approx(expected).epsilon(0.0005));
  }
}

TEST_CASE("gpu_smoothed_envelope handles an empty input without touching the device") {
  sycl::queue q{sycl::property::queue::in_order{}};
  std::vector<int16_t> raw;
  std::vector<float> gpu = dsp::gpu_smoothed_envelope(q, raw, 5);
  CHECK(gpu.empty());
}
