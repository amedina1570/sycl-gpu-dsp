// GPU-executed test for the batched cuFFT-via-SYCL-interop pipeline used by
// stageC_spectrogram.cpp / iq2spectrogram.cpp: framing + Hann window +
// int16 decode (device kernel, sycl_dsp_math.hpp/dsp_math.hpp) -> batched
// cufftExecC2C via AdaptiveCpp_enqueue_custom_operation -> magnitude(dB) +
// fftshift (device kernel). Validated against an independently computed
// reference: the same windowed signal run through the naive DFT
// (dft_lib.hpp, extracted from 03_dft.cpp) instead of cuFFT.
//
// Requires a CUDA-backed SYCL queue; skips (not a failure) if unavailable.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "dsp_math.hpp"
#include "sycl_dsp_math.hpp"
#include "dft_lib.hpp"
#include <cufft.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sycl/sycl.hpp>

namespace {
constexpr int NFFT = 512;

// Runs the production framing+window+cuFFT+magnitude(dB)+fftshift pipeline
// on `batch` frames of a real-valued (Q=0) tone each, returning the
// row-major [frame][fftshifted bin] dB spectrogram.
std::vector<float> run_cufft_pipeline(sycl::queue& q, const std::vector<int>& k0_per_frame) {
  const int nframes = (int)k0_per_frame.size();
  const size_t nsamp = (size_t)NFFT * nframes;

  std::vector<int16_t> raw(nsamp * 2, 0);
  for (int fr = 0; fr < nframes; ++fr)
    for (int n = 0; n < NFFT; ++n) {
      size_t s = (size_t)fr * NFFT + n;
      float tone = std::cos(2.0f * dsp::PI * k0_per_frame[fr] * n / NFFT);
      raw[2*s] = (int16_t)std::lround(32767.0f * tone);
      raw[2*s+1] = 0;
    }

  int16_t* d_raw = sycl::malloc_device<int16_t>(nsamp*2, q);
  cufftComplex* d_batch = sycl::malloc_device<cufftComplex>((size_t)nframes*NFFT, q);
  float* d_spec = sycl::malloc_device<float>((size_t)nframes*NFFT, q);
  q.memcpy(d_raw, raw.data(), nsamp*2*sizeof(int16_t)).wait();

  q.parallel_for(sycl::range<2>{(size_t)nframes, (size_t)NFFT}, [=](sycl::id<2> id) {
    size_t fr = id[0], n = id[1];
    size_t s = fr*NFFT + n;
    float w = dsp::hann_coeff(n, NFFT);
    float i = dsp::decode_i16(d_raw[2*s]);
    float qd = dsp::decode_i16(d_raw[2*s+1]);
    cufftComplex c; c.x = i*w; c.y = qd*w;
    d_batch[fr*NFFT + n] = c;
  }).wait();

  cufftHandle plan;
  if (cufftPlan1d(&plan, NFFT, CUFFT_C2C, nframes) != CUFFT_SUCCESS)
    throw std::runtime_error("cufftPlan1d failed");
  q.submit([&](sycl::handler& cgh) {
    cgh.AdaptiveCpp_enqueue_custom_operation([=](sycl::interop_handle& ih) {
      auto stream = ih.get_native_queue<sycl::backend::cuda>();
      cufftSetStream(plan, stream);
      cufftExecC2C(plan, d_batch, d_batch, CUFFT_FORWARD);
    });
  });

  q.parallel_for(sycl::range<2>{(size_t)nframes, (size_t)NFFT}, [=](sycl::id<2> id) {
    size_t fr = id[0], k = id[1];
    cufftComplex c = d_batch[fr*NFFT + k];
    float p = c.x*c.x + c.y*c.y;
    float db = dsp::db_from_power(p);
    size_t ks = (k + NFFT/2) % NFFT;
    d_spec[fr*NFFT + ks] = db;
  });
  q.wait();

  std::vector<float> spec((size_t)nframes*NFFT);
  q.memcpy(spec.data(), d_spec, (size_t)nframes*NFFT*sizeof(float)).wait();

  cufftDestroy(plan);
  sycl::free(d_raw, q); sycl::free(d_batch, q); sycl::free(d_spec, q);
  return spec;
}

// Same windowing/decode math, but transformed with the naive DFT instead of
// cuFFT, as an independent ground truth. Returns one frame's fftshifted dB
// spectrum.
std::vector<float> reference_frame_spectrum(sycl::queue& q, int k0) {
  std::vector<int16_t> raw(NFFT * 2, 0);
  for (int n = 0; n < NFFT; ++n)
    raw[2*n] = (int16_t)std::lround(32767.0f * std::cos(2.0f * dsp::PI * k0 * n / NFFT));

  std::vector<float> x(NFFT);
  for (int n = 0; n < NFFT; ++n) {
    float w = 0.5f * (1.0f - std::cos(2.0f * dsp::PI * n / (NFFT - 1)));
    x[n] = dsp::decode_i16(raw[2*n]) * w;
  }

  std::vector<float> mag = dsp::naive_dft_mag(q, x);
  std::vector<float> db(NFFT);
  for (int k = 0; k < NFFT; ++k) {
    float p = mag[k]*mag[k];
    size_t ks = (k + NFFT/2) % NFFT;
    db[ks] = 10.0f * std::log10(p + 1e-12f);
  }
  return db;
}

bool cuda_backend_available() {
  sycl::queue q{sycl::property::queue::in_order{}};
  return q.get_device().get_backend() == sycl::backend::cuda;
}
}

TEST_CASE("batched cuFFT pipeline matches an independent naive-DFT reference") {
  if (!cuda_backend_available()) return; // interop requires the CUDA backend

  sycl::queue q{sycl::property::queue::in_order{}};
  std::vector<int> tones = {40, 90};
  std::vector<float> spec = run_cufft_pipeline(q, tones);

  for (size_t fr = 0; fr < tones.size(); ++fr) {
    std::vector<float> ref = reference_frame_spectrum(q, tones[fr]);
    float ref_peak = *std::max_element(ref.begin(), ref.end());

    // cuFFT and the O(N^2) naive DFT accumulate floating-point error in
    // different orders, so far from the tone their sidelobes are dominated
    // by each algorithm's own rounding noise and can disagree by tens of dB
    // despite both being numerically "correct". Only hold bins where the
    // tone itself dominates (within 20 dB of the frame's peak) to a tight
    // tolerance -- that's what actually verifies the batched pipeline
    // reproduces the right magnitude at the right bin.
    for (int k = 0; k < NFFT; ++k) {
      if (ref[k] < ref_peak - 20.0f) continue;
      float actual = spec[fr*NFFT + k];
      CHECK(actual == doctest::Approx(ref[k]).epsilon(0.02));
    }
  }
}

TEST_CASE("batched cuFFT pipeline recovers each frame's tone at its own bin") {
  if (!cuda_backend_available()) return;

  sycl::queue q{sycl::property::queue::in_order{}};
  std::vector<int> tones = {40, 90, 200};
  std::vector<float> spec = run_cufft_pipeline(q, tones);

  for (size_t fr = 0; fr < tones.size(); ++fr) {
    size_t peak = 0;
    for (int k = 0; k < NFFT; ++k)
      if (spec[fr*NFFT+k] > spec[fr*NFFT+peak]) peak = k;

    size_t expect_a = (NFFT/2 + tones[fr]) % NFFT;
    size_t expect_b = (NFFT/2 - tones[fr] + NFFT) % NFFT;
    CAPTURE(fr); CAPTURE(tones[fr]); CAPTURE(peak);
    CHECK((peak == expect_a || peak == expect_b));
  }
}
