/**
 * @file dft_lib.hpp
 * @brief Naive O(N^2) DFT magnitude, as a reusable GPU-executed function.
 *
 * Serves as ground truth for the radix-2 FFT (fft_lib.hpp) in tests, and
 * lets 03_dft.cpp's kernel logic be exercised outside of `main()`.
 */
#pragma once
#include "sycl_util.hpp"
#include <sycl/sycl.hpp>
#include <vector>

namespace dsp {

/// Compute `|DFT(x)|` directly (`O(N^2)`, one GPU thread per output bin, no
/// windowing) -- a correctness baseline, not meant to be fast.
/// @param q Queue to run on.
/// @param x Real-valued input signal, length `N`.
/// @return DFT magnitude, length `N`.
inline std::vector<float> naive_dft_mag(sycl::queue& q, const std::vector<float>& x) {
  const size_t N = x.size();
  constexpr float PI = 3.14159265358979323846f;

  std::vector<float> mag(N, 0.0f);
  float* d_x   = sycl_util::malloc_device_checked<float>(N, q, "d_x");
  float* d_mag = sycl_util::malloc_device_checked<float>(N, q, "d_mag");
  q.memcpy(d_x, x.data(), N * sizeof(float));

  q.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> idx) {
    size_t k = idx[0];
    float re = 0.0f, im = 0.0f;
    for (size_t n = 0; n < N; ++n) {
      float ang = -2.0f * PI * k * n / N;
      re += d_x[n] * sycl::cos(ang);
      im += d_x[n] * sycl::sin(ang);
    }
    d_mag[k] = sycl::sqrt(re * re + im * im);
  });

  q.memcpy(mag.data(), d_mag, N * sizeof(float));
  q.wait();

  sycl::free(d_x, q);
  sycl::free(d_mag, q);
  return mag;
}

} // namespace dsp
