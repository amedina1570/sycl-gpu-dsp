// DSP math callable from SYCL device kernels (uses sycl:: math intrinsics,
// not std::), extracted so the exact formulas used in the production
// kernels (02_window.cpp, stageC_spectrogram.cpp, iq2spectrogram.cpp,
// dedisp.cpp, dmsearch.cpp) can also be exercised by GPU-executed tests.
#pragma once
#include "dsp_constants.hpp"
#include <sycl/sycl.hpp>

namespace dsp {

// w[n] = 0.5 * (1 - cos(2*pi*n / (N-1))), N >= 2.
inline float hann_coeff(size_t n, size_t N) {
  return 0.5f * (1.0f - sycl::cos(2.0f * PI * (float)n / (float)(N - 1)));
}

inline float db_from_power(float power) {
  return 10.0f * sycl::log10(power + POWER_FLOOR);
}

inline float power_from_db(float db) {
  return sycl::pow(10.0f, db * 0.1f);
}

} // namespace dsp
