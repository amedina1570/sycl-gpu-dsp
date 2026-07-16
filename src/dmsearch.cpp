// DM search: sweep DM, dedisperse, measure pulse SNR, find best DM.
// Input: crab_spectrogram.bin (float32 dB, nframes x NFFT).
// Output: dm_snr.bin (pairs of float32: DM, SNR) + best profile.
#include "dsp_math.hpp"
#include "sycl_dsp_math.hpp"
#include "crab_example.hpp"
#include <sycl/sycl.hpp>
#include <vector>
#include <cstdio>
#include <fstream>

int main(int argc, char** argv){
  const char* inpath = (argc>1)? argv[1] : "crab_spectrogram.bin";
  const char* outcurve = (argc>2)? argv[2] : "dm_snr.bin";
  const char* outprof  = (argc>3)? argv[3] : "crab_bestprofile.bin";

  constexpr int NFFT=crab::NFFT, HOP=crab::HOP;
  const double FS=crab::FS, FC=crab::FC, t_frame=HOP/FS;
  const double DM_MIN=crab::DM_MIN, DM_MAX=crab::DM_MAX, DM_STEP=crab::DM_STEP;
  const int nDM=(int)((DM_MAX-DM_MIN)/DM_STEP)+1;

  sycl::queue q{sycl::property::queue::in_order{}};
  printf("Device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

  std::ifstream f(inpath,std::ios::binary|std::ios::ate);
  if(!f){printf("cannot open %s\n",inpath);return 1;}
  std::streamsize bytes=f.tellg(); f.seekg(0);
  size_t nframes=(bytes/sizeof(float))/NFFT;
  printf("nframes=%zu  DM trials=%d\n", nframes, nDM);
  std::vector<float> spec(nframes*NFFT);
  f.read(reinterpret_cast<char*>(spec.data()), bytes);

  // Pre-convert spectrogram dB -> linear power once, on device.
  float* d_spec = sycl::malloc_device<float>(nframes*NFFT, q);
  float* d_lin  = sycl::malloc_device<float>(nframes*NFFT, q);
  float* d_prof = sycl::malloc_device<float>(nframes, q);
  int*   d_cnt  = sycl::malloc_device<int>(nframes, q);
  int*   d_shift= sycl::malloc_device<int>(NFFT, q);
  q.memcpy(d_spec, spec.data(), nframes*NFFT*sizeof(float));
  q.parallel_for(sycl::range<1>{nframes*NFFT}, [=](sycl::id<1> i){
    d_lin[i[0]] = dsp::power_from_db(d_spec[i[0]]);
  }).wait();

  std::vector<float> curve(nDM*2), bestprof(nframes);
  double best_snr=-1; double best_dm=0; std::vector<float> best(nframes);

  for(int t=0;t<nDM;t++){
    double DM=DM_MIN + t*DM_STEP;
    std::vector<int> shift = dsp::compute_dm_shifts(DM, FC, FS, NFFT, t_frame);
    q.memcpy(d_shift, shift.data(), NFFT*sizeof(int));
    q.memset(d_prof,0,nframes*sizeof(float));
    q.memset(d_cnt,0,nframes*sizeof(int)).wait();

    // Sum linear power over valid channels + count valid channels per frame.
    q.parallel_for(sycl::range<1>{nframes}, [=](sycl::id<1> id){
      size_t fr=id[0]; float acc=0; int c=0;
      for(int k=0;k<NFFT;k++){
        long src=(long)fr + d_shift[k];
        if(src>=0 && src<(long)nframes){ acc+=d_lin[src*NFFT+k]; c++; }
      }
      d_prof[fr]=acc; d_cnt[fr]=c;
    }).wait();

    std::vector<float> prof(nframes); std::vector<int> cnt(nframes);
    q.memcpy(prof.data(),d_prof,nframes*sizeof(float)).wait();
    q.memcpy(cnt.data(),d_cnt,nframes*sizeof(int)).wait();

    // Normalize per-frame by valid channel count (mean power per channel).
    for(size_t i=0;i<nframes;i++) prof[i] = cnt[i]>0 ? prof[i]/cnt[i] : 0.0f;

    // SNR = (peak - median) / std, using only frames with full channel coverage.
    std::vector<float> full;
    for(size_t i=0;i<nframes;i++) if(cnt[i]==NFFT) full.push_back(prof[i]);
    if(full.size()<(size_t)dsp::MIN_SNR_FRAMES) continue;
    dsp::SnrStats stats = dsp::compute_snr(full);

    curve[2*t]=(float)DM; curve[2*t+1]=stats.snr;
    if(stats.snr>best_snr){ best_snr=stats.snr; best_dm=DM; best=prof; }
  }

  std::ofstream(outcurve,std::ios::binary)
    .write(reinterpret_cast<char*>(curve.data()), curve.size()*sizeof(float));
  std::ofstream(outprof,std::ios::binary)
    .write(reinterpret_cast<char*>(best.data()), best.size()*sizeof(float));
  printf("BEST DM = %.2f pc/cm^3  (SNR = %.1f)\n", best_dm, best_snr);
  printf("wrote %s and %s\n", outcurve, outprof);

  sycl::free(d_spec,q);sycl::free(d_lin,q);sycl::free(d_prof,q);
  sycl::free(d_cnt,q);sycl::free(d_shift,q);
  return 0;
}
