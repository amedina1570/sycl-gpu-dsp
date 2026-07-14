// Stage A: load ci16_le SigMF IQ, convert to complex float, report stats.
#include "dsp_math.hpp"
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>
#include <fstream>

int main(int argc, char** argv) {
  const char* path = (argc > 1) ? argv[1]
                    : "/home/user/13143544/crab-giantpulse.sigmf-data";
  const double fs = 20e6;      // from metadata

  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) { std::printf("cannot open %s\n", path); return 1; }
  std::streamsize bytes = f.tellg();
  f.seekg(0);

  size_t nsamp = bytes / 4;                 // 2x int16 per complex sample
  std::printf("file bytes = %lld\n", (long long)bytes);
  std::printf("complex samples = %zu  (%.3f s at %.0f Msps)\n",
              nsamp, nsamp / fs, fs/1e6);

  std::vector<int16_t> raw(nsamp * 2);
  f.read(reinterpret_cast<char*>(raw.data()), bytes);

  std::vector<float> re(nsamp), im(nsamp);
  double sumpow = 0.0, maxpow = 0.0; size_t maxidx = 0;
  for (size_t n = 0; n < nsamp; ++n) {
    float i = dsp::decode_i16(raw[2*n]);
    float q = dsp::decode_i16(raw[2*n+1]);
    re[n] = i; im[n] = q;
    double p = double(i)*i + double(q)*q;
    sumpow += p;
    if (p > maxpow) { maxpow = p; maxidx = n; }
  }
  std::printf("mean power = %.6e\n", sumpow / nsamp);
  std::printf("peak power = %.6e at sample %zu (t = %.6f s)\n",
              maxpow, maxidx, maxidx / fs);
  std::printf("first 4 IQ: ");
  for (int k=0;k<4;k++) std::printf("(%.3f,%.3f) ", re[k], im[k]);
  std::printf("\n");
  return 0;
}
