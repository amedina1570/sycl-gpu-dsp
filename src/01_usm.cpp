#include <sycl/sycl.hpp>
#include <vector>
#include <iostream>

int main() {
  constexpr size_t N = 1 << 20;              // ~1M elements

  // in-order queue: kernels/copies run in submission order, no manual deps
  sycl::queue q{sycl::property::queue::in_order{}};
  std::cout << "Device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";

  // Host data
  std::vector<float> h_a(N, 1.0f), h_b(N, 2.0f), h_c(N, 0.0f);

  // Device allocations (device USM — lives in GPU memory)
  float* d_a = sycl::malloc_device<float>(N, q);
  float* d_b = sycl::malloc_device<float>(N, q);
  float* d_c = sycl::malloc_device<float>(N, q);

  // H2D copies
  q.memcpy(d_a, h_a.data(), N * sizeof(float));
  q.memcpy(d_b, h_b.data(), N * sizeof(float));

  // Kernel: raw pointers, no accessors
  q.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> i) {
    d_c[i] = d_a[i] + d_b[i];
  });

  // D2H copy, then wait
  q.memcpy(h_c.data(), d_c, N * sizeof(float));
  q.wait();

  // Verify
  bool ok = (h_c[0] == 3.0f && h_c[N-1] == 3.0f);
  std::cout << "c[0]=" << h_c[0] << "  c[N-1]=" << h_c[N-1]
            << (ok ? "  OK\n" : "  FAIL\n");

  sycl::free(d_a, q);
  sycl::free(d_b, q);
  sycl::free(d_c, q);
  return 0;
}
