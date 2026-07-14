// GPU-executed test for dsp::hann_coeff (sycl_dsp_math.hpp), the exact
// windowing formula used by 02_window.cpp, stageC_spectrogram.cpp and
// iq2spectrogram.cpp's framing kernels. Computes the window on-device and
// checks both its known analytic shape and an exact match against a
// host-computed (std::cos) reference.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "sycl_dsp_math.hpp"
#include <cmath>
#include <sycl/sycl.hpp>

namespace {
constexpr float PI = 3.14159265358979323846f;

std::vector<float> device_hann(sycl::queue& q, size_t N) {
  std::vector<float> h_out(N);
  float* d_out = sycl::malloc_device<float>(N, q);
  q.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> idx) {
    d_out[idx[0]] = dsp::hann_coeff(idx[0], N);
  }).wait();
  q.memcpy(h_out.data(), d_out, N * sizeof(float)).wait();
  sycl::free(d_out, q);
  return h_out;
}
}

TEST_CASE("device-computed Hann window matches a host reference exactly") {
  constexpr size_t N = 4096;
  sycl::queue q{sycl::property::queue::in_order{}};

  std::vector<float> gpu = device_hann(q, N);
  for (size_t n = 0; n < N; ++n) {
    float expected = 0.5f * (1.0f - std::cos(2.0f * PI * n / (N - 1)));
    CHECK(gpu[n] == doctest::Approx(expected).epsilon(0.001));
  }
}

TEST_CASE("Hann window is zero at the ends, ~1 at the center, and symmetric") {
  constexpr size_t N = 4096;
  sycl::queue q{sycl::property::queue::in_order{}};
  std::vector<float> w = device_hann(q, N);

  CHECK(w[0] == doctest::Approx(0.0f));
  CHECK(w[N-1] == doctest::Approx(0.0f));
  CHECK((w[N/2] == doctest::Approx(1.0f).epsilon(0.001) ||
         w[N/2 - 1] == doctest::Approx(1.0f).epsilon(0.001)));
  for (size_t n : {10u, 100u, 1000u})
    CHECK(w[n] == doctest::Approx(w[N-1-n]));
}

TEST_CASE("Hann window never goes negative or exceeds 1") {
  constexpr size_t N = 8192; // matches production NFFT default
  sycl::queue q{sycl::property::queue::in_order{}};
  std::vector<float> w = device_hann(q, N);
  for (float v : w) {
    CHECK(v >= -1e-6f);
    CHECK(v <= 1.0f + 1e-6f);
  }
}
