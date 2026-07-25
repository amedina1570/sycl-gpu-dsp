/**
 * @file dsp_math.hpp
 * @brief Host-only DSP math shared across the dedispersion/DM-search/loader/
 * radar programs.
 *
 * Deliberately free of any SYCL dependency so it can be included (and unit
 * tested) from plain host code such as stageA_load.cpp.
 */
#pragma once
#include "dsp_constants.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace dsp {

/// Normalize a signed 16-bit IQ sample to [-1, 1].
/// @param raw Raw `ci16_le` sample component (I or Q).
/// @return Normalized value in [-1, 1).
inline float decode_i16(int16_t raw) {
  return raw / INT16_FULL_SCALE;
}

/// @name decode_sample overload set
/// So generic (templated) kernels can decode either supported raw sample
/// type with one spelling: int16 gets normalized, float32 is already in
/// natural units and passes through.
///@{
inline float decode_sample(int16_t raw) { return decode_i16(raw); }
inline float decode_sample(float raw)   { return raw; }
///@}

/// Incoherent-dispersion delay, in whole frames, of frequency channel
/// `fk_hz` relative to reference frequency `fref_hz`, for a given
/// dispersion measure. Standard cold-plasma dispersion law:
/// `t(s) = K * DM * (f_GHz^-2)`, K = #DM_DELAY_CONST_S.
/// @param dm Dispersion measure, pc/cm^3.
/// @param fk_hz Channel frequency, Hz.
/// @param fref_hz Reference frequency (delay is zero here), Hz.
/// @param t_frame Frame duration, seconds (used to convert the delay to a
/// whole number of frames).
/// @return Signed frame shift; negative means the channel arrives earlier
/// than the reference.
inline int dm_shift_samples(double dm, double fk_hz, double fref_hz, double t_frame) {
  double fk_ghz = fk_hz / 1e9, fref_ghz = fref_hz / 1e9;
  double delay_s = DM_DELAY_CONST_S * dm * (1.0 / (fk_ghz * fk_ghz) - 1.0 / (fref_ghz * fref_ghz));
  return (int)std::lround(delay_s / t_frame);
}

/// Per-channel frame shifts for an NFFT-wide fftshifted spectrogram, where
/// channel k's frequency is `fc + (k - nfft/2) * fs/nfft` and the reference
/// is the top of the band (k = nfft/2), matching dedisp.cpp/dmsearch.cpp's
/// `f_ref` convention.
/// @param dm Dispersion measure, pc/cm^3.
/// @param fc_hz Center frequency, Hz.
/// @param fs_hz Sample rate, Hz.
/// @param nfft FFT size / number of frequency channels.
/// @param t_frame Frame duration, seconds.
/// @return One shift (see dm_shift_samples()) per channel, length `nfft`.
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

/// Result of compute_snr(): summary statistics of a profile plus its SNR.
struct SnrStats {
  float median = 0.0f;  ///< Median of the input values.
  float mean = 0.0f;    ///< Mean of the input values.
  float stddev = 0.0f;  ///< Population standard deviation of the input values.
  float peak = 0.0f;    ///< Maximum of the input values.
  float snr = 0.0f;     ///< `(peak - median) / stddev`, or 0 if stddev is 0.
};

/// SNR = `(peak - median) / stddev` over a set of per-frame profile values
/// (linear power, mean-per-channel). Matches dmsearch.cpp's scoring.
/// @param values Per-frame profile values; may be empty (returns a
/// zero-initialized SnrStats).
/// @return Populated SnrStats.
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

/// Linear-interpolated percentile, matching `numpy.percentile`'s default
/// (`'linear'`) method.
/// @param values Sample values (taken by value; sorted internally). Empty
/// input returns 0.
/// @param pct Percentile to compute, in [0, 100].
/// @return The interpolated percentile value.
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

/// Result of detect_pulses(): paired rising/falling edges plus the
/// threshold used to find them.
struct PulseDetection {
  std::vector<size_t> rising;   ///< rising[i]: first sample of pulse i.
  std::vector<size_t> falling;  ///< falling[i]: one past pulse i's last sample.
  float threshold = 0.0f;       ///< Envelope threshold used to detect edges.
};

/// Threshold the envelope at `percentile(10) + threshold_frac *
/// (percentile(99.9) - percentile(10))`, and pair rising/falling edges into
/// complete pulses. Pulses cut off at either boundary of `envelope` (no
/// matching edge) are dropped -- their width/PRI can't be measured
/// reliably. Matches view_radar_pulses.py's `detect_pulses`.
/// @param envelope Smoothed amplitude envelope (see radar_lib.hpp's
/// gpu_smoothed_envelope()).
/// @param threshold_frac Fraction between the 10th- and 99.9th-percentile
/// envelope value used as the detection threshold, in (0, 1).
/// @param min_pulse_samples Minimum pulse width, in samples, to keep --
/// shorter detections are discarded as noise.
/// @return Detected pulses (rising/falling sample indices) and the
/// threshold used.
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

/// Result of pulse_stats(): width/PRI/PRF/duty-cycle summary of a pulse train.
struct PulseStats {
  int n_pulses = 0;               ///< Number of detected pulses.
  float pulse_width_mean_s = 0.0f;  ///< Mean pulse width, seconds.
  float pulse_width_std_s = 0.0f;   ///< Pulse width standard deviation, seconds.
  float pri_mean_s = 0.0f;   ///< Mean pulse repetition interval, seconds (see #has_pri).
  float pri_std_s = 0.0f;    ///< PRI standard deviation, seconds (see #has_pri).
  float prf_hz = 0.0f;       ///< Pulse repetition frequency, `1/pri_mean_s` (see #has_pri).
  float duty_cycle = 0.0f;   ///< `pulse_width_mean_s / pri_mean_s` (see #has_pri).
  bool has_pri = false;      ///< False when fewer than 2 pulses (PRI needs a gap between two).
};

/// Pulse width from rising/falling sample indices; PRI from consecutive
/// rising-edge spacing (undefined below 2 pulses); PRF = `1/mean(PRI)`;
/// duty cycle = `mean_width/mean_PRI`. Matches view_radar_pulses.py's
/// `pulse_stats`.
/// @param det Detected pulses, as returned by detect_pulses().
/// @param fs Sample rate, Hz (converts sample counts to seconds).
/// @return Populated PulseStats; `has_pri` is false (and the PRI/PRF/duty
/// fields are left at 0) when fewer than 2 pulses were detected.
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
