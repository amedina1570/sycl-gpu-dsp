#include <sycl/sycl.hpp>
#include <vector>
#include <cmath>
#include <iostream>

// Single-workgroup radix-2 Cooley-Tukey FFT for power-of-two N.
// One workgroup of N/2 work-items cooperates through the whole transform,
// synchronizing between butterfly stages with a local barrier.
int main() {
  constexpr size_t N    = 1024;                 // power of two
  constexpr int    LOGN = 10;                    // log2(N)
  constexpr float  PI   = 3.14159265358979323846f;
  constexpr int    K0   = 64;                    // input tone bin

  sycl::queue q{sycl::property::queue::in_order{}};
  std::cout << "Device: "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";

  std::vector<float> h_re(N), h_im(N, 0.0f), h_mag(N, 0.0f);
  for (size_t n = 0; n < N; ++n)
    h_re[n] = std::cos(2.0f * PI * K0 * n / N);

  float* d_re  = sycl::malloc_device<float>(N, q);
  float* d_im  = sycl::malloc_device<float>(N, q);
  float* d_mag = sycl::malloc_device<float>(N, q);
  q.memcpy(d_re, h_re.data(), N * sizeof(float));
  q.memcpy(d_im, h_im.data(), N * sizeof(float));

  size_t local = N / 2;
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 1> sre(sycl::range<1>{N}, h);
    sycl::local_accessor<float, 1> sim(sycl::range<1>{N}, h);

    h.parallel_for(
      sycl::nd_range<1>{sycl::range<1>{local}, sycl::range<1>{local}},
      [=](sycl::nd_item<1> it) {
        size_t tid = it.get_local_id(0);

        // Load with bit-reversal permutation (each item moves 2 elements)
        for (int pass = 0; pass < 2; ++pass) {
          size_t i = 2 * tid + pass;
          size_t r = 0, x = i;
          for (int b = 0; b < LOGN; ++b) { r = (r << 1) | (x & 1); x >>= 1; }
          sre[r] = d_re[i];
          sim[r] = d_im[i];
        }
        it.barrier(sycl::access::fence_space::local_space);

        // log2(N) butterfly stages
        for (int s = 1; s <= LOGN; ++s) {
          size_t m  = (size_t)1 << s;
          size_t mh = m >> 1;
          size_t grp = tid / mh;
          size_t pos = tid % mh;
          size_t j   = grp * m + pos;
          size_t k   = j + mh;

          float ang = -2.0f * PI * pos / m;
          float wr = sycl::cos(ang), wi = sycl::sin(ang);

          float tr = wr * sre[k] - wi * sim[k];
          float ti = wr * sim[k] + wi * sre[k];
          float ur = sre[j], ui = sim[j];
          sre[j] = ur + tr;  sim[j] = ui + ti;
          sre[k] = ur - tr;  sim[k] = ui - ti;

          it.barrier(sycl::access::fence_space::local_space);
        }

        // Write magnitudes
        for (int pass = 0; pass < 2; ++pass) {
          size_t i = 2 * tid + pass;
          d_mag[i] = sycl::sqrt(sre[i]*sre[i] + sim[i]*sim[i]);
        }
      });
  });

  q.memcpy(h_mag.data(), d_mag, N * sizeof(float));
  q.wait();

  size_t peak = 0;
  for (size_t k = 0; k < N; ++k)
    if (h_mag[k] > h_mag[peak]) peak = k;

  std::cout << "peak bin = " << peak
            << "  (expected " << K0 << " or " << (N - K0) << ")\n";
  std::cout << "mag[" << K0 << "] = " << h_mag[K0]
            << "   mag[0] = " << h_mag[0] << "\n";
  bool ok = (peak == (size_t)K0) || (peak == N - (size_t)K0);
  std::cout << (ok ? "OK\n" : "FAIL\n");

  sycl::free(d_re, q); sycl::free(d_im, q); sycl::free(d_mag, q);
  return 0;
}
