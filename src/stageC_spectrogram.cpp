// Stage C: Crab-pulse SigMF -> windowed batched cuFFT -> spectrogram (float32 .bin)
#include "sycl_dsp_math.hpp"
#include "dsp_math.hpp"
#include "crab_example.hpp"
#include "cufft_interop.hpp"
#include <sycl/sycl.hpp>
#include <cufft.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <cmath>
#include <cstdio>
#include <fstream>

int main(int argc, char** argv) {
  const char* inpath  = (argc>1)? argv[1] : "crab-giantpulse.sigmf-data";
  const char* outpath = (argc>2)? argv[2] : "crab_spectrogram.bin";

  constexpr int NFFT = crab::NFFT, HOP = crab::HOP;

  sycl::queue q = cufft_util::make_inorder_queue();
  if (!cufft_util::require_cuda_backend(q)) return 1;

  // --- Load ci16_le ---
  std::ifstream f(inpath, std::ios::binary | std::ios::ate);
  if(!f){ fprintf(stderr, "cannot open %s\n", inpath); return 1; }
  std::streamsize bytes = f.tellg(); f.seekg(0);
  size_t nsamp = bytes/4;
  if (nsamp < (size_t)NFFT) {
    fprintf(stderr, "file too short for NFFT=%d\n", NFFT);
    return 1;
  }
  std::vector<int16_t> raw(nsamp*2);
  // Read only whole samples: `bytes` may include a trailing partial sample
  // (e.g. a truncated file) that the buffer above deliberately excludes.
  f.read(reinterpret_cast<char*>(raw.data()), nsamp*2*sizeof(int16_t));
  if(!f){ fprintf(stderr, "short read from %s\n", inpath); return 1; }
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
  cufftResult fft_status = CUFFT_SUCCESS;
  cufft_util::enqueue_exec_c2c_forward(q, plan, d_batch, &fft_status);

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
  if (fft_status != CUFFT_SUCCESS) {
    fprintf(stderr, "cuFFT error %d executing FFT\n", fft_status);
    return 1;
  }

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
