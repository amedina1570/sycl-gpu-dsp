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

// Linear-interpolated percentile, matching numpy.percentile's default
// ('linear') method. pct in [0, 100].
inline float percentile(std::vector<float> values, double pct) {
  if (values.empty()) return 0.0f;
  std::sort(values.begin(), values.end());
  if (values.size() == 1) return values[0];
  double idx = (pct / 100.0) * (double)(values.size() - 1);
  size_t lo = (size_t)std::floor(idx);
  size_t hi = (size_t)std::ceil(idx);
  double frac = idx - (double)lo;
  return (float)(values[lo] + frac * (values[hi] - values[lo]));
}

struct PulseDetection {
  std::vector<size_t> rising, falling;  // rising[i]: first sample of pulse i;
                                         // falling[i]: one past its last sample
  float threshold = 0.0f;
};

// Threshold the envelope at percentile(10) + threshold_frac * (percentile(99.9)
// - percentile(10)), and pair rising/falling edges into complete pulses.
// Pulses cut off at either boundary of `envelope` (no matching edge) are
// dropped -- their width/PRI can't be measured reliably. Matches
// view_radar_pulses.py's detect_pulses.
inline PulseDetection detect_pulses(const std::vector<float>& envelope,
                                     float threshold_frac, size_t min_pulse_samples) {
  PulseDetection out;
  if (envelope.size() < 2) return out;

  float floor = percentile(envelope, 10.0);
  float peak  = percentile(envelope, 99.9);
  out.threshold = floor + threshold_frac * (peak - floor);

  std::vector<size_t> rising, falling;
  bool prev_above = envelope[0] > out.threshold;
  for (size_t i = 1; i < envelope.size(); ++i) {
    bool above = envelope[i] > out.threshold;
    if (above && !prev_above) rising.push_back(i);
    else if (!above && prev_above) falling.push_back(i);
    prev_above = above;
  }

  // A falling edge with no preceding rising edge belongs to a pulse that was
  // already active at the start of `envelope`; a trailing rising edge with
  // no following falling edge belongs to a pulse still active at the end.
  // Both are cut off and can't be measured, so drop them.
  if (!falling.empty() && (rising.empty() || falling.front() < rising.front()))
    falling.erase(falling.begin());
  if (!rising.empty() && (falling.empty() || rising.back() > falling.back()))
    rising.pop_back();

  size_t n = std::min(rising.size(), falling.size());
  for (size_t i = 0; i < n; ++i) {
    if (falling[i] - rising[i] >= min_pulse_samples) {
      out.rising.push_back(rising[i]);
      out.falling.push_back(falling[i]);
    }
  }
  return out;
}

struct PulseStats {
  int n_pulses = 0;
  float pulse_width_mean_s = 0.0f, pulse_width_std_s = 0.0f;
  float pri_mean_s = 0.0f, pri_std_s = 0.0f, prf_hz = 0.0f, duty_cycle = 0.0f;
  bool has_pri = false;  // false when fewer than 2 pulses (PRI needs a gap)
};

// Pulse width from rising/falling sample indices; PRI from consecutive
// rising-edge spacing (undefined below 2 pulses); PRF = 1/mean(PRI); duty
// cycle = mean_width/mean_PRI. Matches view_radar_pulses.py's pulse_stats.
inline PulseStats pulse_stats(const PulseDetection& det, double fs) {
  PulseStats out;
  out.n_pulses = (int)det.rising.size();
  if (out.n_pulses == 0) return out;

  std::vector<float> widths(out.n_pulses);
  for (int i = 0; i < out.n_pulses; ++i)
    widths[i] = (float)((double)(det.falling[i] - det.rising[i]) / fs);

  double wmean = 0.0;
  for (float w : widths) wmean += w;
  wmean /= widths.size();
  out.pulse_width_mean_s = (float)wmean;

  double wvar = 0.0;
  for (float w : widths) wvar += (w - wmean) * (w - wmean);
  wvar /= widths.size();
  out.pulse_width_std_s = (float)std::sqrt(wvar);

  if (out.n_pulses >= 2) {
    std::vector<double> pri(out.n_pulses - 1);
    for (int i = 0; i + 1 < out.n_pulses; ++i)
      pri[i] = (double)(det.rising[i + 1] - det.rising[i]) / fs;

    double pmean = 0.0;
    for (double p : pri) pmean += p;
    pmean /= pri.size();

    double pvar = 0.0;
    for (double p : pri) pvar += (p - pmean) * (p - pmean);
    pvar /= pri.size();

    out.has_pri = true;
    out.pri_mean_s = (float)pmean;
    out.pri_std_s = (float)std::sqrt(pvar);
    out.prf_hz = (float)(1.0 / pmean);
    out.duty_cycle = (float)(wmean / pmean);
  }
  return out;
}

} // namespace dsp
