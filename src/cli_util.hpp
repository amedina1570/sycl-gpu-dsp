/**
 * @file cli_util.hpp
 * @brief Pure validation/defaulting helpers for the iq2spectrogram CLI.
 *
 * Split out of `main()` so they can be unit tested without going through
 * `argv`/`exit()`.
 */
#pragma once
#include "dsp_constants.hpp"
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>

/// @namespace cli
/// @brief CLI argument validation/defaulting and chunk-sizing math shared by
/// the streaming pipeline (iq2spectrogram.cpp).
namespace cli {

/// Device bytes one FFT bin costs per frame in the streaming pipeline's
/// batch buffer (`sizeof(cufftComplex)`).
constexpr std::size_t BATCH_BYTES_PER_BIN = 8;
/// Device bytes one FFT bin costs per frame in the streaming pipeline's
/// spectrogram buffer (`sizeof(float)`).
constexpr std::size_t SPEC_BYTES_PER_BIN  = 4;

/// @param value Value to check.
/// @return True if `value` is a positive power of two.
inline bool is_power_of_two(int value) {
  return value > 0 && (value & (value - 1)) == 0;
}

/// Resolve an unset hop size to 75% overlap.
/// @param nfft FFT size.
/// @param hop Requested hop, or 0 to mean "unset".
/// @return `hop` unchanged if nonzero, else `nfft / dsp::DEFAULT_HOP_DIVISOR`.
inline int resolve_hop(int nfft, int hop) {
  return hop == 0 ? nfft / dsp::DEFAULT_HOP_DIVISOR : hop;
}

/// @param datatype Candidate SigMF datatype string.
/// @return True if `datatype` is `"ci16_le"` or `"cf32_le"`.
inline bool is_supported_datatype(const std::string& datatype) {
  return datatype == "ci16_le" || datatype == "cf32_le";
}

/// Parse a complete floating-point CLI value.
/// @param text Candidate value.
/// @return Parsed value, or std::nullopt if the whole string is not one
/// finite number.
inline std::optional<double> parse_double(const std::string& text) {
  if (text.empty()) return std::nullopt;
  errno = 0;
  char* end = nullptr;
  double value = std::strtod(text.c_str(), &end);
  if (end == text.c_str() || *end != '\0' || errno == ERANGE || !std::isfinite(value))
    return std::nullopt;
  return value;
}

/// Parse a complete single-precision floating-point CLI value.
/// @param text Candidate value.
/// @return Parsed value, or std::nullopt if invalid/out of range.
inline std::optional<float> parse_float(const std::string& text) {
  auto value = parse_double(text);
  if (!value || *value < -std::numeric_limits<float>::max() ||
      *value > std::numeric_limits<float>::max())
    return std::nullopt;
  return (float)*value;
}

/// Parse a complete long integer CLI value.
/// @param text Candidate value.
/// @return Parsed value, or std::nullopt if invalid/out of range.
inline std::optional<long> parse_long(const std::string& text) {
  if (text.empty()) return std::nullopt;
  errno = 0;
  char* end = nullptr;
  long value = std::strtol(text.c_str(), &end, 10);
  if (end == text.c_str() || *end != '\0' || errno == ERANGE)
    return std::nullopt;
  return value;
}

/// Parse a complete int CLI value.
/// @param text Candidate value.
/// @return Parsed value, or std::nullopt if invalid/out of range.
inline std::optional<int> parse_int(const std::string& text) {
  auto value = parse_long(text);
  if (!value || *value < std::numeric_limits<int>::min() ||
      *value > std::numeric_limits<int>::max())
    return std::nullopt;
  return (int)*value;
}

/// Bytes per complex IQ sample on disk for a supported SigMF datatype.
/// @param datatype `"ci16_le"` or `"cf32_le"` (see is_supported_datatype()).
/// @return #dsp::BYTES_PER_CI16_SAMPLE or #dsp::BYTES_PER_CF32_SAMPLE.
inline std::size_t bytes_per_complex_sample(const std::string& datatype) {
  return datatype == "ci16_le" ? dsp::BYTES_PER_CI16_SAMPLE
                               : dsp::BYTES_PER_CF32_SAMPLE;
}

/// How many frames to process per GPU chunk so the batched cuFFT
/// pipeline's two dominant device allocations (the `cufftComplex` batch
/// buffer and the `float` spectrogram buffer) fit within `chunk_mb`
/// megabytes.
/// @param chunk_mb Target GPU memory budget per chunk, in megabytes.
/// @param nfft FFT size.
/// @param total_frames Total frame count in the file (an upper clamp: a
/// generous budget never returns more frames than actually exist).
/// @return Frame count per chunk, clamped to `[1, total_frames]`.
inline std::size_t resolve_chunk_frames(long chunk_mb, int nfft, std::size_t total_frames) {
  const std::size_t bytes_per_frame =
      (std::size_t)nfft * (BATCH_BYTES_PER_BIN + SPEC_BYTES_PER_BIN);
  const std::size_t budget = (std::size_t)chunk_mb * 1024ull * 1024ull;
  std::size_t frames = budget / bytes_per_frame;
  if (frames < 1) frames = 1;
  if (frames > total_frames) frames = total_frames;
  return frames;
}

/// Complex samples needed to cover `frames_in_chunk` hop-spaced NFFT-wide
/// frames, i.e. the read size for one chunk of the framing kernel.
/// @param frames_in_chunk Number of frames in the chunk.
/// @param hop Hop size, in samples.
/// @param nfft FFT size (width of the last frame).
/// @return `(frames_in_chunk - 1) * hop + nfft`.
inline std::size_t chunk_sample_count(std::size_t frames_in_chunk, int hop, int nfft) {
  return (frames_in_chunk - 1) * (std::size_t)hop + (std::size_t)nfft;
}

/// Absolute raw-sample index a chunk starting at frame `f0` must seek to:
/// frame `f` reads samples starting at `f*hop`, so the chunk's first frame
/// reads starting at `f0*hop`.
///
/// Pulled out as its own unit-tested function because an
/// off-by-a-few-samples error here is invisible downstream -- for
/// stationary signals a fixed sample offset is a pure time shift, and the
/// Fourier shift theorem means it doesn't move a magnitude spectrum, so
/// this exact bug class can silently pass any spectrogram-output-based
/// check.
/// @param f0 Index of the chunk's first frame.
/// @param hop Hop size, in samples.
/// @return `f0 * hop`.
inline std::size_t chunk_start_sample(std::size_t f0, int hop) {
  return f0 * (std::size_t)hop;
}

} // namespace cli
