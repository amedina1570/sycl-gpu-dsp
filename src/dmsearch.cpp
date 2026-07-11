// DM search: sweep DM, dedisperse, measure pulse SNR, find best DM.
// Input: crab_spectrogram.bin (float32 dB, nframes x NFFT).
// Output: dm_snr.bin (pairs of float32: DM, SNR) + best profile.
#include <sycl/sycl.hpp>
#include <vector>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <algorithm>

int main(int argc, char** argv){
  const char* inpath = (argc>1)? argv[1] : "/home/user/crab_spectrogram.bin";
  const char* outcurve = "/home/user/dm_snr.bin";
  const char* outprof  = "/home/user/crab_bestprofile.bin";

  constexpr int NFFT=8192, HOP=2048;
  const double FS=20e6, FC=410e6, t_frame=HOP/FS;
  const double DM_MIN=40.0, DM_MAX=75.0, DM_STEP=0.25;
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
    d_lin[i[0]] = sycl::pow(10.0f, d_spec[i[0]]*0.1f);
  }).wait();

  double f_ref=(FC+(NFFT/2)*(double)FS/NFFT);
  std::vector<int> shift(NFFT);
  std::vector<float> curve(nDM*2), bestprof(nframes);
  double best_snr=-1; double best_dm=0; std::vector<float> best(nframes);

  for(int t=0;t<nDM;t++){
    double DM=DM_MIN + t*DM_STEP;
    for(int k=0;k<NFFT;k++){
      double fk=(FC+(k-NFFT/2)*(double)FS/NFFT), fkG=fk/1e9, frG=f_ref/1e9;
      double d=4.148808e-3*DM*(1.0/(fkG*fkG)-1.0/(frG*frG));
      shift[k]=(int)lround(d/t_frame);
    }
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
    if(full.size()<10) continue;
    std::vector<float> s=full; std::sort(s.begin(),s.end());
    float med=s[s.size()/2];
    double mean=0; for(float v:full) mean+=v; mean/=full.size();
    double var=0; for(float v:full) var+=(v-mean)*(v-mean); var/=full.size();
    float sd=std::sqrt((float)var);
    float peak=*std::max_element(full.begin(),full.end());
    float snr = sd>0 ? (peak-med)/sd : 0.0f;

    curve[2*t]=(float)DM; curve[2*t+1]=snr;
    if(snr>best_snr){ best_snr=snr; best_dm=DM; best=prof; }
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
