// GPU-executed tests: the hand-written radix-2 FFT (fft_lib.hpp, extracted
// from 04_fft.cpp) cross-checked against the naive O(N^2) DFT (dft_lib.hpp,
// extracted from 03_dft.cpp), which is the same validation strategy the
// original programs used informally (single-tone peak-bin checks), now
// formalized as per-bin numeric comparisons across several signals/sizes.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "dft_lib.hpp"
#include "fft_lib.hpp"
#include <cmath>
#include <sycl/sycl.hpp>

namespace {
constexpr float PI = 3.14159265358979323846f;

sycl::queue& test_queue() {
  static sycl::queue q{sycl::property::queue::in_order{}};
  return q;
}

std::vector<float> cosine_tone(size_t N, int k0) {
  std::vector<float> x(N);
  for (size_t n = 0; n < N; ++n) x[n] = std::cos(2.0f * PI * k0 * n / N);
  return x;
}
}

TEST_CASE("radix2 FFT matches naive DFT bin-for-bin on a single tone") {
  constexpr size_t N = 1024;
  constexpr int K0 = 64;
  sycl::queue& q = test_queue();

  std::vector<float> x = cosine_tone(N, K0);
  std::vector<float> dft_mag = dsp::naive_dft_mag(q, x);
  std::vector<float> fft_mag = dsp::radix2_fft_mag(q, x, std::vector<float>(N, 0.0f));

  REQUIRE(dft_mag.size() == fft_mag.size());
  for (size_t k = 0; k < N; ++k)
    CHECK(fft_mag[k] == doctest::Approx(dft_mag[k]).epsilon(0.01));
}

TEST_CASE("radix2 FFT matches naive DFT on a multi-tone signal") {
  constexpr size_t N = 512;
  sycl::queue& q = test_queue();

  std::vector<float> x(N);
  for (size_t n = 0; n < N; ++n)
    x[n] = std::cos(2.0f*PI*30*n/N) + 0.5f*std::cos(2.0f*PI*100*n/N);

  std::vector<float> dft_mag = dsp::naive_dft_mag(q, x);
  std::vector<float> fft_mag = dsp::radix2_fft_mag(q, x, std::vector<float>(N, 0.0f));

  for (size_t k = 0; k < N; ++k)
    CHECK(fft_mag[k] == doctest::Approx(dft_mag[k]).epsilon(0.02));
}

TEST_CASE("radix2 FFT recovers the correct peak bin for several tone/size combinations") {
  sycl::queue& q = test_queue();
  struct Case { size_t N; int K0; };
  for (Case c : {Case{256,17}, Case{1024,64}, Case{2048,300}}) {
    std::vector<float> x = cosine_tone(c.N, c.K0);
    std::vector<float> mag = dsp::radix2_fft_mag(q, x, std::vector<float>(c.N, 0.0f));

    size_t peak = 0;
    for (size_t k = 0; k < c.N; ++k) if (mag[k] > mag[peak]) peak = k;

    CAPTURE(c.N); CAPTURE(c.K0);
    CHECK((peak == (size_t)c.K0 || peak == c.N - (size_t)c.K0));
  }
}

TEST_CASE("DFT of a zero signal is zero everywhere") {
  constexpr size_t N = 256;
  sycl::queue& q = test_queue();
  std::vector<float> mag = dsp::naive_dft_mag(q, std::vector<float>(N, 0.0f));
  for (float m : mag) CHECK(m == doctest::Approx(0.0f));
}

TEST_CASE("DFT of a DC signal has all energy in bin 0") {
  constexpr size_t N = 256;
  sycl::queue& q = test_queue();
  std::vector<float> mag = dsp::naive_dft_mag(q, std::vector<float>(N, 1.0f));
  CHECK(mag[0] == doctest::Approx((float)N));
  for (size_t k = 1; k < N; ++k) CHECK(mag[k] < 1e-2f);
}
