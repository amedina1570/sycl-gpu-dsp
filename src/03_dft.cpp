#include "dft_lib.hpp"
#include <sycl/sycl.hpp>
#include <vector>
#include <cmath>
#include <iostream>

int main() {
  constexpr size_t N  = 1024;                // samples / bins
  constexpr float  PI = 3.14159265358979323846f;
  constexpr int    K0 = 64;                  // tone at bin 64 (integer bin -> clean)

  // Async exceptions (e.g. a failed copy/kernel) are logged instead of being
  // silently dropped -- q.wait() alone does not rethrow them.
  sycl::queue q{
      [](sycl::exception_list exceptions) {
        for (const std::exception_ptr& e : exceptions) {
          try {
            std::rethrow_exception(e);
          } catch (const sycl::exception& ex) {
            std::cerr << "asynchronous SYCL exception: " << ex.what() << "\n";
          }
        }
      },
      sycl::property::queue::in_order{}};
  std::cout << "Device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";

  // Real input: cosine at exactly bin K0 => x[n] = cos(2*pi*K0*n/N)
  std::vector<float> h_x(N);
  for (size_t n = 0; n < N; ++n)
    h_x[n] = std::cos(2.0f * PI * K0 * n / N);

  // X[k] = sum_n x[n] * exp(-i 2*pi*k*n/N); mag[k] = sqrt(Re^2 + Im^2)
  std::vector<float> h_mag = dsp::naive_dft_mag(q, h_x);

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

  return ok ? 0 : 1;
}
