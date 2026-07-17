// Incoherent dedispersion of the Crab spectrogram on GPU (SYCL).
// Input: crab_spectrogram.bin (float32, nframes x NFFT, dB).
// Output: dedispersed spectrogram + 1D pulse profile (sum over freq).
#include "dsp_math.hpp"
#include "sycl_dsp_math.hpp"
#include "crab_example.hpp"
#include <sycl/sycl.hpp>
#include <cstdint>
#include <vector>
#include <cstdio>
#include <fstream>

int main(int argc, char** argv) {
  const char* inpath  = (argc>1)? argv[1] : "crab_spectrogram.bin";
  const char* outspec = (argc>2)? argv[2] : "crab_dedispersed.bin";
  const char* outprof = (argc>3)? argv[3] : "crab_profile.bin";

  constexpr int NFFT = crab::NFFT, HOP = crab::HOP;
  const double FS = crab::FS, FC = crab::FC;
  const double DM = crab::DM;                 // pc/cm^3 (Crab); tune this
  const double t_frame = HOP / FS;

  sycl::queue q{sycl::property::queue::in_order{}};
  printf("Device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

  // Load spectrogram
  std::ifstream f(inpath, std::ios::binary | std::ios::ate);
  if(!f){ fprintf(stderr, "cannot open %s\n", inpath); return 1; }
  std::streamsize bytes = f.tellg(); f.seekg(0);
  size_t nframes = (bytes/sizeof(float)) / NFFT;
  printf("nframes = %zu\n", nframes);
  std::vector<float> spec(nframes*NFFT);
  // Read only whole frames: `bytes` may include a trailing partial frame
  // (e.g. a truncated file) that the buffer above deliberately excludes.
  f.read(reinterpret_cast<char*>(spec.data()), nframes*NFFT*sizeof(float));
  if(!f){ fprintf(stderr, "short read from %s\n", inpath); return 1; }

  // Precompute per-bin frame shifts on host (small, NFFT entries)
  std::vector<int> shift = dsp::compute_dm_shifts(DM, FC, FS, NFFT, t_frame);

  float* d_spec  = sycl::malloc_device<float>(nframes*NFFT, q);
  float* d_dedis = sycl::malloc_device<float>(nframes*NFFT, q);
  float* d_prof  = sycl::malloc_device<float>(nframes, q);
  int*   d_shift = sycl::malloc_device<int>(NFFT, q);
  q.memcpy(d_spec, spec.data(), nframes*NFFT*sizeof(float));
  q.memcpy(d_shift, shift.data(), NFFT*sizeof(int));
  q.memset(d_prof, 0, nframes*sizeof(float)).wait();

  // Dedisperse: dedis[fr,k] = spec[fr + shift[k], k]  (pull earlier-arriving
  // low-freq energy back to align with the top of band)
  q.parallel_for(sycl::range<2>{nframes, (size_t)NFFT}, [=](sycl::id<2> id){
    size_t fr = id[0]; int k = (int)id[1];
    long src = (long)fr + d_shift[k];
    float v = (src >= 0 && src < (long)nframes) ? d_spec[src*NFFT + k]
                                                : dsp::SPECTROGRAM_FLOOR_DB;
    d_dedis[fr*NFFT + k] = v;
  });

  // Pulse profile: sum (in linear power) across frequency for each frame.
  // spec is in dB, so convert back: p = 10^(db/10).
  q.parallel_for(sycl::range<1>{nframes}, [=](sycl::id<1> id){
    size_t fr = id[0]; float acc = 0.0f;
    for(int k=0;k<NFFT;k++){
      float db = d_dedis[fr*NFFT + k];
      acc += dsp::power_from_db(db);
    }
    d_prof[fr] = acc;
  });
  q.wait();

  std::vector<float> dedis(nframes*NFFT), prof(nframes);
  q.memcpy(dedis.data(), d_dedis, nframes*NFFT*sizeof(float)).wait();
  q.memcpy(prof.data(),  d_prof,  nframes*sizeof(float)).wait();

  std::ofstream(outspec, std::ios::binary)
      .write(reinterpret_cast<char*>(dedis.data()), nframes*NFFT*sizeof(float));
  std::ofstream(outprof, std::ios::binary)
      .write(reinterpret_cast<char*>(prof.data()), nframes*sizeof(float));

  // Report profile peak (should be near the pulse arrival at top of band)
  size_t pk=0; for(size_t i=0;i<nframes;i++) if(prof[i]>prof[pk]) pk=i;
  printf("wrote %s and %s\n", outspec, outprof);
  printf("profile peak at frame %zu (t=%.4f s)  DM=%.1f\n", pk, pk*t_frame, DM);

  sycl::free(d_spec,q); sycl::free(d_dedis,q);
  sycl::free(d_prof,q); sycl::free(d_shift,q);
  return 0;
}
