// Naive O(N^2) DFT magnitude, as a reusable GPU-executed function so it can
// serve as ground truth for the radix-2 FFT (fft_lib.hpp) in tests, and so
// 03_dft.cpp's kernel logic can be exercised outside of main().
#pragma once
#include <sycl/sycl.hpp>
#include <vector>

namespace dsp {

inline std::vector<float> naive_dft_mag(sycl::queue& q, const std::vector<float>& x) {
  const size_t N = x.size();
  constexpr float PI = 3.14159265358979323846f;

  std::vector<float> mag(N, 0.0f);
  float* d_x   = sycl::malloc_device<float>(N, q);
  float* d_mag = sycl::malloc_device<float>(N, q);
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
