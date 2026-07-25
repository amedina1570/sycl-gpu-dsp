// GPU-executed amplitude-envelope + boxcar-smoothing kernels, extracted so
// radar_pulses.cpp's exact kernel logic can be exercised outside of main()
// (e.g. by GPU-executed tests), mirroring dft_lib.hpp/fft_lib.hpp.
#pragma once
#include "dsp_math.hpp"
#include "sycl_util.hpp"
#include <sycl/sycl.hpp>
#include <vector>

namespace dsp {

// raw holds interleaved I,Q pairs (raw.size() == 2*nsamp); Raw is int16_t
// (ci16_le, normalized via decode_sample) or float (cf32_le, passed through).
// Returns the smoothed envelope, one value per complex sample.
//
// Smoothing matches numpy's np.convolve(envelope, ones(w)/w, mode='same')
// exactly: output[n] averages envelope[n+off-w+1 .. n+off] (off = (w-1)/2),
// treating samples outside [0, nsamp) as zero -- i.e. it tapers near the
// edges rather than shrinking the averaging window, same as the reference
// Python implementation in view_radar_pulses.py.
template <typename Raw>
inline std::vector<float> gpu_smoothed_envelope(sycl::queue& q,
                                                  const std::vector<Raw>& raw,
                                                  int smooth_samples) {
  const size_t nsamp = raw.size() / 2;
  std::vector<float> smooth(nsamp, 0.0f);
  if (nsamp == 0) return smooth;

  Raw*   d_raw    = sycl_util::malloc_device_checked<Raw>(raw.size(), q, "d_raw");
  float* d_env    = sycl_util::malloc_device_checked<float>(nsamp, q, "d_env");
  float* d_smooth = sycl_util::malloc_device_checked<float>(nsamp, q, "d_smooth");
  q.memcpy(d_raw, raw.data(), raw.size() * sizeof(Raw));

  // Envelope: magnitude of the complex I/Q sample.
  q.parallel_for(sycl::range<1>{nsamp}, [=](sycl::id<1> idx) {
    size_t n = idx[0];
    float i = dsp::decode_sample(d_raw[2 * n]);
    float qd = dsp::decode_sample(d_raw[2 * n + 1]);
    d_env[n] = sycl::hypot(i, qd);
  });

  // smooth_samples <= 0 degenerates to a no-op (window of 1).
  const int w = smooth_samples > 0 ? smooth_samples : 1;
  const long off = (long)((w - 1) / 2);
  q.parallel_for(sycl::range<1>{nsamp}, [=](sycl::id<1> idx) {
    long n = (long)idx[0];
    float acc = 0.0f;
    for (int d = 0; d < w; ++d) {
      long m = n + off - w + 1 + d;
      if (m >= 0 && m < (long)nsamp) acc += d_env[m];
    }
    d_smooth[n] = acc / (float)w;
  });

  q.memcpy(smooth.data(), d_smooth, nsamp * sizeof(float));
  q.wait();

  sycl::free(d_raw, q);
  sycl::free(d_env, q);
  sycl::free(d_smooth, q);
  return smooth;
}

} // namespace dsp
