// Stage B: cuFFT via AdaptiveCpp interop, synthetic data, in isolation.
#include <sycl/sycl.hpp>
#include <cufft.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <cstdio>
#include <exception>

#define CUFFT_CHECK(x) do { cufftResult r=(x); \
  if(r!=CUFFT_SUCCESS){printf("cuFFT error %d at line %d\n",r,__LINE__);return 1;} }while(0)

int main() {
  constexpr int   NFFT  = 8192;
  constexpr int   BATCH = 4;
  constexpr float PI    = 3.14159265358979323846f;
  constexpr int   K0    = 100;

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

  q.submit([&](sycl::handler& cgh) {
    cgh.AdaptiveCpp_enqueue_custom_operation([=](sycl::interop_handle& ih) {
      auto stream = ih.get_native_queue<sycl::backend::cuda>();
      cufftResult r = cufftSetStream(plan, stream);
      if (r != CUFFT_SUCCESS) {
        std::printf("cuFFT error %d in cufftSetStream\n", r);
        return;
      }
      r = cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
      if (r != CUFFT_SUCCESS) {
        std::printf("cuFFT error %d in cufftExecC2C\n", r);
      }
    });
  });
  q.wait();

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
