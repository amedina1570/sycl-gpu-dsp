// Stage C: Crab-pulse SigMF -> windowed batched cuFFT -> spectrogram (float32 .bin)
#include "sycl_dsp_math.hpp"
#include "dsp_math.hpp"
#include "crab_example.hpp"
#include <sycl/sycl.hpp>
#include <cufft.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <exception>

#define CUFFT_CHECK(x) do { cufftResult r=(x); \
  if(r!=CUFFT_SUCCESS){printf("cuFFT error %d at line %d\n",r,__LINE__);return 1;} }while(0)

int main(int argc, char** argv) {
  const char* inpath  = (argc>1)? argv[1] : "crab-giantpulse.sigmf-data";
  const char* outpath = (argc>2)? argv[2] : "crab_spectrogram.bin";

  constexpr int NFFT = crab::NFFT, HOP = crab::HOP;

  sycl::queue q{
    [](sycl::exception_list exceptions) {
      for (const std::exception_ptr& e : exceptions) {
        try {
          std::rethrow_exception(e);
        } catch (const sycl::exception& ex) {
          std::printf("asynchronous SYCL exception: %s\n", ex.what());
        }
      }
    },
    sycl::property::queue::in_order{}};
  printf("Device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
  if (q.get_device().get_backend() != sycl::backend::cuda) {
    printf("cuFFT interop requires a SYCL queue backed by the CUDA backend\n");
    return 1;
  }

  // --- Load ci16_le ---
  std::ifstream f(inpath, std::ios::binary | std::ios::ate);
  if(!f){ printf("cannot open %s\n", inpath); return 1; }
  std::streamsize bytes = f.tellg(); f.seekg(0);
  size_t nsamp = bytes/4;
  if (nsamp < (size_t)NFFT) {
    printf("file too short for NFFT=%d\n", NFFT);
    return 1;
  }
  std::vector<int16_t> raw(nsamp*2);
  f.read(reinterpret_cast<char*>(raw.data()), bytes);
  size_t nframes = (nsamp - NFFT)/HOP + 1;
  printf("samples=%zu  frames=%zu\n", nsamp, nframes);

  // --- Device buffers ---
  int16_t*      d_raw   = sycl::malloc_device<int16_t>(nsamp*2, q);
  cufftComplex* d_batch = sycl::malloc_device<cufftComplex>(nframes*NFFT, q);
  float*        d_spec  = sycl::malloc_device<float>(nframes*NFFT, q);
  q.memcpy(d_raw, raw.data(), nsamp*2*sizeof(int16_t)).wait();

  // --- Framing + int16->float + Hann window (fused), 2D range ---
  q.parallel_for(sycl::range<2>{nframes, (size_t)NFFT}, [=](sycl::id<2> id){
    size_t fr = id[0], n = id[1];
    size_t s  = fr*HOP + n;                       // source sample
    float w = dsp::hann_coeff(n, NFFT);
    float i = dsp::decode_i16(d_raw[2*s]);
    float qd= dsp::decode_i16(d_raw[2*s+1]);
    cufftComplex c; c.x = i*w; c.y = qd*w;
    d_batch[fr*NFFT + n] = c;
  });

  // --- Batched cuFFT via interop ---
  cufftHandle plan;
  CUFFT_CHECK(cufftPlan1d(&plan, NFFT, CUFFT_C2C, (int)nframes));
  q.submit([&](sycl::handler& cgh){
    cgh.AdaptiveCpp_enqueue_custom_operation([=](sycl::interop_handle& ih){
      auto stream = ih.get_native_queue<sycl::backend::cuda>();
      cufftResult r = cufftSetStream(plan, stream);
      if (r != CUFFT_SUCCESS) {
        printf("cuFFT error %d in cufftSetStream\n", r);
        return;
      }
      r = cufftExecC2C(plan, d_batch, d_batch, CUFFT_FORWARD);
      if (r != CUFFT_SUCCESS) {
        printf("cuFFT error %d in cufftExecC2C\n", r);
      }
    });
  });

  // --- Magnitude (dB) + fftshift, 2D range ---
  q.parallel_for(sycl::range<2>{nframes, (size_t)NFFT}, [=](sycl::id<2> id){
    size_t fr = id[0], k = id[1];
    cufftComplex c = d_batch[fr*NFFT + k];
    float p = c.x*c.x + c.y*c.y;
    float db = dsp::db_from_power(p);
    size_t ks = (k + NFFT/2) % NFFT;              // fftshift: DC -> center
    d_spec[fr*NFFT + ks] = db;
  });
  q.wait();

  // --- Copy back and write float32 .bin (row-major: frames x NFFT) ---
  std::vector<float> spec(nframes*NFFT);
  q.memcpy(spec.data(), d_spec, nframes*NFFT*sizeof(float)).wait();
  std::ofstream out(outpath, std::ios::binary);
  out.write(reinterpret_cast<char*>(spec.data()), nframes*NFFT*sizeof(float));
  printf("wrote %s  (%zu frames x %d bins, float32)\n", outpath, nframes, NFFT);

  cufftDestroy(plan);
  sycl::free(d_raw,q); sycl::free(d_batch,q); sycl::free(d_spec,q);
  return 0;
}
