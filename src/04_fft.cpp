#include "fft_lib.hpp"
#include <sycl/sycl.hpp>
#include <vector>
#include <cmath>
#include <iostream>

// Single-workgroup radix-2 Cooley-Tukey FFT for power-of-two N.
// One workgroup of N/2 work-items cooperates through the whole transform,
// synchronizing between butterfly stages with a local barrier.
int main() {
  constexpr size_t N    = 1024;                 // power of two
  constexpr float  PI   = 3.14159265358979323846f;
  constexpr int    K0   = 64;                    // input tone bin

  sycl::queue q{sycl::property::queue::in_order{}};
  std::cout << "Device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";

  std::vector<float> h_re(N), h_im(N, 0.0f);
  for (size_t n = 0; n < N; ++n)
    h_re[n] = std::cos(2.0f * PI * K0 * n / N);

  std::vector<float> h_mag = dsp::radix2_fft_mag(q, h_re, h_im);

  size_t peak = 0;
  for (size_t k = 0; k < N; ++k)
    if (h_mag[k] > h_mag[peak]) peak = k;

  std::cout << "peak bin = " << peak
            << "  (expected " << K0 << " or " << (N - K0) << ")\n";
  std::cout << "mag[" << K0 << "] = " << h_mag[K0]
            << "   mag[0] = " << h_mag[0] << "\n";
  bool ok = (peak == (size_t)K0) || (peak == N - (size_t)K0);
  std::cout << (ok ? "OK\n" : "FAIL\n");

  return 0;
}
