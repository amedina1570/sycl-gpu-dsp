#include <sycl/sycl.hpp>
#include <vector>
#include <cmath>
#include <iostream>

int main() {
  constexpr size_t N  = 1024;                // samples / bins
  constexpr float  PI = 3.14159265358979323846f;
  constexpr int    K0 = 64;                  // tone at bin 64 (integer bin -> clean)

  sycl::queue q{sycl::property::queue::in_order{}};
  std::cout << "Device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";

  // Real input: cosine at exactly bin K0 => x[n] = cos(2*pi*K0*n/N)
  std::vector<float> h_x(N);
  for (size_t n = 0; n < N; ++n)
    h_x[n] = std::cos(2.0f * PI * K0 * n / N);

  std::vector<float> h_mag(N, 0.0f);

  float* d_x   = sycl::malloc_device<float>(N, q);
  float* d_mag = sycl::malloc_device<float>(N, q);
  q.memcpy(d_x, h_x.data(), N * sizeof(float));

  // One work-item per output bin k. Each sums over all n:
  //   X[k] = sum_n x[n] * exp(-i 2*pi*k*n/N)
  //   mag[k] = sqrt(Re^2 + Im^2)
  q.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> idx) {
    size_t k = idx[0];
    float re = 0.0f, im = 0.0f;
    for (size_t n = 0; n < N; ++n) {
      float ang = -2.0f * PI * k * n / N;
      float s = sycl::sin(ang);
      float c = sycl::cos(ang);
      re += d_x[n] * c;
      im += d_x[n] * s;
    }
    d_mag[k] = sycl::sqrt(re*re + im*im);
  });

  q.memcpy(h_mag.data(), d_mag, N * sizeof(float));
  q.wait();

  // Find the peak bin
  size_t peak = 0;
  for (size_t k = 0; k < N; ++k)
    if (h_mag[k] > h_mag[peak]) peak = k;

  std::cout << "peak bin = " << peak
            << "  (expected " << K0 << " or " << (N - K0) << ")\n";
  std::cout << "mag[" << K0 << "] = " << h_mag[K0]
            << "   mag[0] = " << h_mag[0] << "\n";

  bool ok = (peak == (size_t)K0) || (peak == N - (size_t)K0);
  std::cout << (ok ? "OK\n" : "FAIL\n");

  sycl::free(d_x, q);
  sycl::free(d_mag, q);
  return 0;
}
