/**
 * @file dsp_constants.hpp
 * @brief Shared numeric constants for the DSP pipeline.
 *
 * Deliberately free of any SYCL or CUDA dependency (plain `constexpr`), so
 * the same values can be used from host code, SYCL device kernels, and the
 * unit tests. This is the single definition site for numbers that were
 * previously duplicated as literals across the loader, spectrogram,
 * dedispersion, and DM-search programs.
 */
#pragma once
#include <cstddef>

/// @namespace dsp
/// @brief Host- and device-shared DSP math, constants, and detection/stats
/// logic (no SYCL dependency here; see sycl_dsp_math.hpp for the
/// device-kernel-only counterparts that need `sycl::` math intrinsics).
namespace dsp {

/// Pi in single precision, for device-side trig in the FFT/window kernels.
constexpr float PI = 3.14159265358979323846f;

/// Full-scale magnitude of a signed 16-bit sample. Dividing a ci16 IQ
/// sample by this maps [-32768, 32767] into roughly [-1, 1).
constexpr float INT16_FULL_SCALE = 32768.0f;

/// Small power floor added before log10 so db_from_power(0) is a finite,
/// very negative dB value instead of -inf.
constexpr float POWER_FLOOR = 1e-12f;

/// dB value written for spectrogram cells that have no valid source
/// sample, e.g. dedispersion shifts that fall off the ends of the recording.
constexpr float SPECTROGRAM_FLOOR_DB = -120.0f;

/// Cold-plasma dispersion delay constant K in
/// \f[ \Delta t = K \cdot DM \cdot f_{GHz}^{-2} \f]
/// with the dispersion measure DM in pc/cm^3 and frequency \f$f\f$ in GHz.
/// K = 4.148808e-3 is the standard \f$1/(2.41 \times 10^{-4}\,\mathrm{MHz})\f$
/// coefficient expressed for GHz and seconds.
constexpr double DM_DELAY_CONST_S = 4.148808e-3;

/// Default STFT FFT size (power of two) for the spectrogram pipeline.
constexpr int DEFAULT_NFFT = 8192;
/// Default STFT hop divisor: hop = nfft/DEFAULT_HOP_DIVISOR => 75% overlap.
constexpr int DEFAULT_HOP_DIVISOR = 4;

/// Minimum number of fully-covered frames required before a DM trial is
/// scored, so an SNR isn't computed from a handful of edge frames.
constexpr int MIN_SNR_FRAMES = 10;

/// Bytes per complex IQ sample for SigMF `ci16_le` (2 x int16).
constexpr std::size_t BYTES_PER_CI16_SAMPLE = 4;
/// Bytes per complex IQ sample for SigMF `cf32_le` (2 x float32).
constexpr std::size_t BYTES_PER_CF32_SAMPLE = 8;

} // namespace dsp
