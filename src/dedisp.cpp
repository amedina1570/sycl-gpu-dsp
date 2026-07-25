// Incoherent dedispersion of the Crab spectrogram on GPU (SYCL).
// Input: crab_spectrogram.bin (float32, nframes x NFFT, dB).
// Output: dedispersed spectrogram + 1D pulse profile (sum over freq).
#include "dsp_math.hpp"
#include "sycl_dsp_math.hpp"
#include "crab_example.hpp"
#include "sycl_util.hpp"
#include <sycl/sycl.hpp>
#include <cstdint>
#include <vector>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <string>

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

  // This pipeline allocates device buffers sized to the whole spectrogram --
  // fine for the small Crab dataset it's hardcoded to, but a large input
  // (e.g. iq2spectrogram's output on a real multi-GB recording) would
  // silently blow past GPU memory. Fail with a clear message instead of a
  // cryptic OOM/segfault. SYCL has no portable "free" memory query, so this
  // checks against total device memory rather than currently-free memory.
  {
    size_t total_bytes = q.get_device().get_info<sycl::info::device::global_mem_size>();
    size_t device_bytes_needed = 2 * nframes*(size_t)NFFT*sizeof(float)  // d_spec + d_dedis
                                + nframes*sizeof(float)                  // d_prof
                                + (size_t)NFFT*sizeof(int);               // d_shift
    if (device_bytes_needed > (size_t)(total_bytes * 0.9)) {
      fprintf(stderr,
        "dedisp: this file needs ~%.1f GB of GPU memory (device has ~%.1f GB total) -- "
        "this pedagogical pipeline loads the whole spectrogram as one unbounded buffer.\n",
        device_bytes_needed / 1e9, total_bytes / 1e9);
      return 1;
    }
  }

  float* d_spec  = sycl_util::malloc_device_checked<float>(nframes*NFFT, q, "d_spec");
  float* d_dedis = sycl_util::malloc_device_checked<float>(nframes*NFFT, q, "d_dedis");
  float* d_prof  = sycl_util::malloc_device_checked<float>(nframes, q, "d_prof");
  int*   d_shift = sycl_util::malloc_device_checked<int>(NFFT, q, "d_shift");
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

  // Sidecar JSON so view_dedisp.py can read NFFT/HOP/FS/FC/DM instead of
  // hardcoding a second copy of crab_example.hpp's values.
  {
    std::string outspec_s = outspec;
    auto pos = outspec_s.rfind(".bin");
    std::string meta_path = (pos != std::string::npos) ? outspec_s.substr(0, pos) + ".json"
                                                         : outspec_s + ".json";
    std::ofstream jf(meta_path);
    jf << std::setprecision(17)
       << "{\n"
       << "  \"nfft\": " << NFFT << ",\n"
       << "  \"hop\": "  << HOP  << ",\n"
       << "  \"fs\": "   << FS   << ",\n"
       << "  \"fc\": "   << FC   << ",\n"
       << "  \"dm\": "   << DM   << "\n"
       << "}\n";
  }

  // Report profile peak (should be near the pulse arrival at top of band)
  size_t pk=0; for(size_t i=0;i<nframes;i++) if(prof[i]>prof[pk]) pk=i;
  printf("wrote %s and %s\n", outspec, outprof);
  printf("profile peak at frame %zu (t=%.4f s)  DM=%.1f\n", pk, pk*t_frame, DM);

  sycl::free(d_spec,q); sycl::free(d_dedis,q);
  sycl::free(d_prof,q); sycl::free(d_shift,q);
  return 0;
}
