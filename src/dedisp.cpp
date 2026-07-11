// Incoherent dedispersion of the Crab spectrogram on GPU (SYCL).
// Input: crab_spectrogram.bin (float32, nframes x NFFT, dB).
// Output: dedispersed spectrogram + 1D pulse profile (sum over freq).
#include <sycl/sycl.hpp>
#include <cstdint>
#include <vector>
#include <cmath>
#include <cstdio>
#include <fstream>

int main(int argc, char** argv) {
  const char* inpath  = (argc>1)? argv[1] : "/home/user/crab_spectrogram.bin";
  const char* outspec = "/home/user/crab_dedispersed.bin";
  const char* outprof = "/home/user/crab_profile.bin";

  constexpr int NFFT = 8192, HOP = 2048;
  const double FS = 20e6, FC = 410e6;
  const double DM = 56.7;                    // pc/cm^3 (Crab); tune this
  const double t_frame = HOP / FS;

  sycl::queue q{sycl::property::queue::in_order{}};
  printf("Device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

  // Load spectrogram
  std::ifstream f(inpath, std::ios::binary | std::ios::ate);
  if(!f){ printf("cannot open %s\n", inpath); return 1; }
  std::streamsize bytes = f.tellg(); f.seekg(0);
  size_t nframes = (bytes/sizeof(float)) / NFFT;
  printf("nframes = %zu\n", nframes);
  std::vector<float> spec(nframes*NFFT);
  f.read(reinterpret_cast<char*>(spec.data()), bytes);

  // Precompute per-bin frame shifts on host (small, NFFT entries)
  double f_ref = (FC + (NFFT/2)*FS/NFFT);    // top of band, Hz
  std::vector<int> shift(NFFT);
  for(int k=0;k<NFFT;k++){
    double fk = (FC + (k - NFFT/2)*(double)FS/NFFT);   // Hz (fftshifted axis)
    double fkG=fk/1e9, frG=f_ref/1e9;
    double d = 4.148808e-3*DM*(1.0/(fkG*fkG) - 1.0/(frG*frG)); // s
    shift[k] = (int)lround(d / t_frame);
  }

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
    float v = (src >= 0 && src < (long)nframes) ? d_spec[src*NFFT + k] : -120.0f;
    d_dedis[fr*NFFT + k] = v;
  });

  // Pulse profile: sum (in linear power) across frequency for each frame.
  // spec is in dB, so convert back: p = 10^(db/10).
  q.parallel_for(sycl::range<1>{nframes}, [=](sycl::id<1> id){
    size_t fr = id[0]; float acc = 0.0f;
    for(int k=0;k<NFFT;k++){
      float db = d_dedis[fr*NFFT + k];
      acc += sycl::pow(10.0f, db*0.1f);
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
