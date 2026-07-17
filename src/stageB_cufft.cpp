// Stage B: cuFFT via AdaptiveCpp interop, synthetic data, in isolation.
#include "cufft_interop.hpp"
#include <sycl/sycl.hpp>
#include <cufft.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <cstdio>

int main() {
  constexpr int   NFFT  = 8192;
  constexpr int   BATCH = 4;
  constexpr float PI    = 3.14159265358979323846f;
  constexpr int   K0    = 100;

  sycl::queue q = cufft_util::make_inorder_queue();
  if (!cufft_util::require_cuda_backend(q)) return 1;

  const size_t total = size_t(NFFT) * BATCH;
  cufftComplex* d_data = sycl::malloc_device<cufftComplex>(total, q);

  std::vector<cufftComplex> h(total);
  for (int b = 0; b < BATCH; ++b)
    for (int n = 0; n < NFFT; ++n) {
      h[b*NFFT + n].x = cosf(2.0f*PI*K0*n/NFFT);
      h[b*NFFT + n].y = 0.0f;
    }
  q.memcpy(d_data, h.data(), total*sizeof(cufftComplex)).wait();

  cufftHandle plan;
  CUFFT_CHECK(cufftPlan1d(&plan, NFFT, CUFFT_C2C, BATCH));

  cufftResult fft_status = CUFFT_SUCCESS;
  cufft_util::enqueue_exec_c2c_forward(q, plan, d_data, &fft_status);
  q.wait();
  if (fft_status != CUFFT_SUCCESS) {
    fprintf(stderr, "cuFFT error %d executing FFT\n", fft_status);
    return 1;
  }

  q.memcpy(h.data(), d_data, total*sizeof(cufftComplex)).wait();
  int peak = 0; float pmax = 0.0f;
  for (int k = 0; k < NFFT; ++k) {
    float m = h[k].x*h[k].x + h[k].y*h[k].y;
    if (m > pmax) { pmax = m; peak = k; }
  }
  printf("row0 peak bin = %d (expected %d or %d)\n", peak, K0, NFFT-K0);
  printf("%s\n", (peak==K0 || peak==NFFT-K0) ? "OK" : "FAIL");

  cufftDestroy(plan);
  sycl::free(d_data, q);
  return 0;
}
