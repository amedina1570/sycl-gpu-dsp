// Pure validation/defaulting helpers for the iq2spectrogram CLI, split out
// of main() so they can be unit tested without going through argv/exit().
#pragma once
#include "dsp_constants.hpp"
#include <cstddef>
#include <string>

namespace cli {

// Device bytes each FFT bin costs per frame in the streaming pipeline: one
// cufftComplex in the batch buffer plus one float in the spectrogram buffer.
constexpr std::size_t BATCH_BYTES_PER_BIN = 8;   // sizeof(cufftComplex)
constexpr std::size_t SPEC_BYTES_PER_BIN  = 4;   // sizeof(float)

inline bool is_power_of_two(int value) {
  return value > 0 && (value & (value - 1)) == 0;
}

// hop==0 means "unset" -> default to 75% overlap (nfft / DEFAULT_HOP_DIVISOR).
inline int resolve_hop(int nfft, int hop) {
  return hop == 0 ? nfft / dsp::DEFAULT_HOP_DIVISOR : hop;
}

inline bool is_supported_datatype(const std::string& datatype) {
  return datatype == "ci16_le" || datatype == "cf32_le";
}

// Bytes per complex IQ sample on disk for a supported SigMF datatype.
inline std::size_t bytes_per_complex_sample(const std::string& datatype) {
  return datatype == "ci16_le" ? dsp::BYTES_PER_CI16_SAMPLE
                               : dsp::BYTES_PER_CF32_SAMPLE;
}

// How many frames to process per GPU chunk so the batched cuFFT pipeline's
// two dominant device allocations (the cufftComplex batch buffer and the
// float spectrogram buffer) fit within chunk_mb megabytes. Clamped to
// [1, total_frames].
inline std::size_t resolve_chunk_frames(long chunk_mb, int nfft, std::size_t total_frames) {
  const std::size_t bytes_per_frame =
      (std::size_t)nfft * (BATCH_BYTES_PER_BIN + SPEC_BYTES_PER_BIN);
  const std::size_t budget = (std::size_t)chunk_mb * 1024ull * 1024ull;
  std::size_t frames = budget / bytes_per_frame;
  if (frames < 1) frames = 1;
  if (frames > total_frames) frames = total_frames;
  return frames;
}

// Complex samples needed to cover `frames_in_chunk` hop-spaced NFFT-wide
// frames, i.e. the read size for one chunk of the framing kernel.
inline std::size_t chunk_sample_count(std::size_t frames_in_chunk, int hop, int nfft) {
  return (frames_in_chunk - 1) * (std::size_t)hop + (std::size_t)nfft;
}

} // namespace cli
