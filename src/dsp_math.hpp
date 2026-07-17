// Host-only DSP math shared across the dedispersion/DM-search/loader
// programs. Deliberately free of any SYCL dependency so it can be included
// (and unit tested) from plain host code such as stageA_load.cpp.
#pragma once
#include "dsp_constants.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace dsp {

// Normalize a signed 16-bit IQ sample pair to [-1, 1] floats.
inline float decode_i16(int16_t raw) {
  return raw / INT16_FULL_SCALE;
}

// Overload set so generic (templated) kernels can decode either supported
// raw sample type with one spelling: int16 gets normalized, float32 is
// already in natural units and passes through.
inline float decode_sample(int16_t raw) { return decode_i16(raw); }
inline float decode_sample(float raw)   { return raw; }

// Incoherent-dispersion delay (in frames) of frequency channel fk_hz
// relative to reference frequency fref_hz, for a given dispersion measure.
// Standard cold-plasma dispersion law: t(s) = K * DM * (f_GHz^-2), K = DM_DELAY_CONST_S.
inline int dm_shift_samples(double dm, double fk_hz, double fref_hz, double t_frame) {
  double fk_ghz = fk_hz / 1e9, fref_ghz = fref_hz / 1e9;
  double delay_s = DM_DELAY_CONST_S * dm * (1.0 / (fk_ghz * fk_ghz) - 1.0 / (fref_ghz * fref_ghz));
  return (int)std::lround(delay_s / t_frame);
}

// Per-channel frame shifts for an NFFT-wide fftshifted spectrogram, where
// channel k's frequency is fc + (k - nfft/2) * fs/nfft and the reference is
// the top of the band (k = nfft/2, i.e. fc + (nfft/2)*fs/nfft... matching
// dedisp.cpp/dmsearch.cpp's f_ref convention).
inline std::vector<int> compute_dm_shifts(double dm, double fc_hz, double fs_hz,
                                           int nfft, double t_frame) {
  double fref_hz = fc_hz + (nfft / 2) * fs_hz / nfft;
  std::vector<int> shift(nfft);
  for (int k = 0; k < nfft; ++k) {
    double fk_hz = fc_hz + (k - nfft / 2) * fs_hz / nfft;
    shift[k] = dm_shift_samples(dm, fk_hz, fref_hz, t_frame);
  }
  return shift;
}

struct SnrStats {
  float median = 0.0f;
  float mean = 0.0f;
  float stddev = 0.0f;
  float peak = 0.0f;
  float snr = 0.0f;
};

// SNR = (peak - median) / stddev over a set of per-frame profile values
// (linear power, mean-per-channel). Matches dmsearch.cpp's scoring.
inline SnrStats compute_snr(const std::vector<float>& values) {
  SnrStats out;
  if (values.empty()) return out;

  std::vector<float> sorted = values;
  std::sort(sorted.begin(), sorted.end());
  out.median = sorted[sorted.size() / 2];

  double mean = 0.0;
  for (float v : values) mean += v;
  mean /= values.size();
  out.mean = (float)mean;

  double var = 0.0;
  for (float v : values) var += (v - mean) * (v - mean);
  var /= values.size();
  out.stddev = std::sqrt((float)var);

  out.peak = *std::max_element(values.begin(), values.end());
  out.snr = out.stddev > 0 ? (out.peak - out.median) / out.stddev : 0.0f;
  return out;
}

} // namespace dsp
