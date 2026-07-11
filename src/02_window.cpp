#include <sycl/sycl.hpp>
#include <vector>
#include <cmath>
#include <iostream>

int main() {
  constexpr size_t N = 4096;                 // frame length
  constexpr float PI = 3.14159265358979323846f;

  sycl::queue q{sycl::property::queue::in_order{}};
  std::cout << "Device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";

  // Host: a test signal (all ones) so the output IS the window shape
  std::vector<float> h_sig(N, 1.0f);
  std::vector<float> h_out(N, 0.0f);

  float* d_sig = sycl::malloc_device<float>(N, q);
  float* d_out = sycl::malloc_device<float>(N, q);

  q.memcpy(d_sig, h_sig.data(), N * sizeof(float));

  // Hann window:  w[n] = 0.5 * (1 - cos(2*pi*n / (N-1)))
  // Each work-item computes its own coefficient from its global index.
  q.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> idx) {
    size_t n = idx[0];
    float w = 0.5f * (1.0f - sycl::cos(2.0f * PI * n / (N - 1)));
    d_out[n] = d_sig[n] * w;
  });

  q.memcpy(h_out.data(), d_out, N * sizeof(float));
  q.wait();

  // Checks: Hann is 0 at the ends, 1 at the center, symmetric.
  auto approx = [](float a, float b){ return std::fabs(a - b) < 1e-4f; };
  bool ends_zero   = approx(h_out[0], 0.0f) && approx(h_out[N-1], 0.0f);
  bool center_one  = approx(h_out[N/2], 1.0f) || approx(h_out[N/2 - 1], 1.0f);
  bool symmetric   = approx(h_out[10], h_out[N-1-10]);

  std::cout << "w[0]="      << h_out[0]
            << "  w[N/2]="  << h_out[N/2]
            << "  w[N-1]="  << h_out[N-1] << "\n";
  std::cout << "ends_zero=" << ends_zero
            << "  center_one=" << center_one
            << "  symmetric=" << symmetric
            << ((ends_zero && center_one && symmetric) ? "  OK\n" : "  FAIL\n");

  sycl::free(d_sig, q);
  sycl::free(d_out, q);
  return 0;
}
