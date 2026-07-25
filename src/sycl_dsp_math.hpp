/**
 * @file sycl_dsp_math.hpp
 * @brief DSP math callable from SYCL device kernels (uses `sycl::` math
 * intrinsics, not `std::`).
 *
 * Extracted so the exact formulas used in the production kernels
 * (02_window.cpp, stageC_spectrogram.cpp, iq2spectrogram.cpp, dedisp.cpp,
 * dmsearch.cpp) can also be exercised by GPU-executed tests.
 */
#pragma once
#include "dsp_constants.hpp"
#include <sycl/sycl.hpp>

namespace dsp {

/// Hann window coefficient: `w[n] = 0.5 * (1 - cos(2*pi*n / (N-1)))`.
/// @param n Sample index, `0 <= n < N`.
/// @param N Window length; must be >= 2.
/// @return Window coefficient in [0, 1].
inline float hann_coeff(size_t n, size_t N) {
  return 0.5f * (1.0f - sycl::cos(2.0f * PI * (float)n / (float)(N - 1)));
}

/// Convert linear power to dB, with a small floor (#POWER_FLOOR) added
/// first so a zero-power input gives a finite (very negative) result
/// instead of `-inf`.
/// @param power Linear power (e.g. `|X[k]|^2`).
/// @return `10 * log10(power + POWER_FLOOR)`.
inline float db_from_power(float power) {
  return 10.0f * sycl::log10(power + POWER_FLOOR);
}

/// Inverse of db_from_power() (ignoring the floor, which is negligible at
/// any dB value the floor itself wouldn't dominate).
/// @param db Power in dB.
/// @return Linear power, `10^(db/10)`.
inline float power_from_db(float db) {
  return sycl::pow(10.0f, db * 0.1f);
}

} // namespace dsp
