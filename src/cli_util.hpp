// Pure validation/defaulting helpers for the iq2spectrogram CLI, split out
// of main() so they can be unit tested without going through argv/exit().
#pragma once
#include <cstddef>
#include <string>

namespace cli {

inline bool is_power_of_two(int value) {
  return value > 0 && (value & (value - 1)) == 0;
}

// hop==0 means "unset" -> default to 75% overlap (nfft/4).
inline int resolve_hop(int nfft, int hop) {
  return hop == 0 ? nfft / 4 : hop;
}

inline bool is_supported_datatype(const std::string& datatype) {
  return datatype == "ci16_le" || datatype == "cf32_le";
}

// How many frames to process per GPU chunk so the batched cuFFT pipeline's
// two dominant device allocations (the cufftComplex batch buffer and the
// float spectrogram buffer, 8 and 4 bytes/sample respectively) fit within
// chunk_mb megabytes. Clamped to [1, total_frames].
inline std::size_t resolve_chunk_frames(long chunk_mb, int nfft, std::size_t total_frames) {
  const std::size_t bytes_per_frame = (std::size_t)nfft * (8 + 4);
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
