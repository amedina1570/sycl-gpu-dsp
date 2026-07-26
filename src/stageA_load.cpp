/**
 * @file stageA_load.cpp
 * @brief Stage A: load `ci16_le` SigMF IQ, convert to complex float,
 * report stats.
 *
 * Pedagogical build-up step (fixed to the Crab dataset, single-shot
 * full-file load). For real files, iq2spectrogram.cpp supersedes this --
 * it does the same load/decode plus streamed chunking so files larger
 * than host/GPU memory don't crash. See docs/TUTORIAL.md §2.1.
 */
#include "dsp_math.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>
#include <fstream>

int main(int argc, char** argv) {
  const char* path = (argc > 1) ? argv[1] : "crab-giantpulse.sigmf-data";
  const double fs = 20e6;      // from metadata

  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) { std::fprintf(stderr, "cannot open %s\n", path); return 1; }
  std::streamsize bytes = f.tellg();
  f.seekg(0);

  size_t nsamp = bytes / 4;                 // 2x int16 per complex sample
  std::printf("file bytes = %lld\n", (long long)bytes);
  std::printf("complex samples = %zu  (%.3f s at %.0f Msps)\n",
              nsamp, nsamp / fs, fs/1e6);

  std::vector<int16_t> raw(nsamp * 2);
  // Read only whole samples: `bytes` may include a trailing partial sample
  // that the buffer above deliberately excludes.
  f.read(reinterpret_cast<char*>(raw.data()), nsamp * 2 * sizeof(int16_t));
  if (!f) { std::fprintf(stderr, "short read from %s\n", path); return 1; }

  constexpr size_t NSHOW = 4;               // leading samples echoed below
  float show[NSHOW][2] = {};
  double sumpow = 0.0, maxpow = 0.0; size_t maxidx = 0;
  for (size_t n = 0; n < nsamp; ++n) {
    float i = dsp::decode_i16(raw[2*n]);
    float q = dsp::decode_i16(raw[2*n+1]);
    if (n < NSHOW) { show[n][0] = i; show[n][1] = q; }
    double p = double(i)*i + double(q)*q;
    sumpow += p;
    if (p > maxpow) { maxpow = p; maxidx = n; }
  }
  std::printf("mean power = %.6e\n", sumpow / nsamp);
  std::printf("peak power = %.6e at sample %zu (t = %.6f s)\n",
              maxpow, maxidx, maxidx / fs);
  std::printf("first %zu IQ: ", std::min(NSHOW, nsamp));
  for (size_t k = 0; k < std::min(NSHOW, nsamp); ++k)
    std::printf("(%.3f,%.3f) ", show[k][0], show[k][1]);
  std::printf("\n");
  return 0;
}
