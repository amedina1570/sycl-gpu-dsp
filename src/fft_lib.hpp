/**
 * @file fft_lib.hpp
 * @brief Single-workgroup radix-2 Cooley-Tukey FFT magnitude, as a reusable
 * GPU-executed function.
 *
 * Extracted from 04_fft.cpp so its exact kernel logic can be unit tested
 * (e.g. cross-checked against dft_lib.hpp's naive DFT).
 */
#pragma once
#include "sycl_util.hpp"
#include <sycl/sycl.hpp>
#include <cmath>
#include <vector>

namespace dsp {

/// Compute `|FFT(re_in + i*im_in)|` via radix-2 Cooley-Tukey, entirely
/// within one work-group's local memory (bit-reversal permutation, then
/// `log2(N)` butterfly stages, each separated by a barrier).
/// @param q Queue to run on.
/// @param re_in Real part of the input signal, length `N`.
/// @param im_in Imaginary part of the input signal, length `N`.
/// @return FFT magnitude, length `N`.
/// @note `N` must be a power of two, and `N/2` must not exceed the
/// device's max work-group size (this single workgroup does the whole
/// transform) -- there's no multi-work-group decomposition here. See
/// cufft_interop.hpp for the production FFT path, which has neither
/// limitation.
inline std::vector<float> radix2_fft_mag(sycl::queue& q,
                                          const std::vector<float>& re_in,
                                          const std::vector<float>& im_in) {
  const size_t N = re_in.size();
  const int LOGN = (int)std::lround(std::log2((double)N));
  constexpr float PI = 3.14159265358979323846f;

  std::vector<float> mag(N, 0.0f);
  float* d_re  = sycl_util::malloc_device_checked<float>(N, q, "d_re");
  float* d_im  = sycl_util::malloc_device_checked<float>(N, q, "d_im");
  float* d_mag = sycl_util::malloc_device_checked<float>(N, q, "d_mag");
  q.memcpy(d_re, re_in.data(), N * sizeof(float));
  q.memcpy(d_im, im_in.data(), N * sizeof(float));

  size_t local = N / 2;
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 1> sre(sycl::range<1>{N}, h);
    sycl::local_accessor<float, 1> sim(sycl::range<1>{N}, h);

    h.parallel_for(
      sycl::nd_range<1>{sycl::range<1>{local}, sycl::range<1>{local}},
      [=](sycl::nd_item<1> it) {
        size_t tid = it.get_local_id(0);

        for (int pass = 0; pass < 2; ++pass) {
          size_t i = 2 * tid + pass;
          size_t r = 0, x = i;
          for (int b = 0; b < LOGN; ++b) { r = (r << 1) | (x & 1); x >>= 1; }
          sre[r] = d_re[i];
          sim[r] = d_im[i];
        }
        it.barrier(sycl::access::fence_space::local_space);

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

        for (int pass = 0; pass < 2; ++pass) {
          size_t i = 2 * tid + pass;
          d_mag[i] = sycl::sqrt(sre[i]*sre[i] + sim[i]*sim[i]);
        }
      });
  });

  q.memcpy(mag.data(), d_mag, N * sizeof(float));
  q.wait();

  sycl::free(d_re, q); sycl::free(d_im, q); sycl::free(d_mag, q);
  return mag;
}

} // namespace dsp
