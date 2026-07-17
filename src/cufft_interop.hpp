// Shared plumbing for driving cuFFT from SYCL (AdaptiveCpp) code: the
// in-order queue with async-exception logging, the CUDA-backend check, and
// the interop submission that runs a batched in-place C2C FFT on the queue's
// native CUDA stream. Extracted from stageB_cufft.cpp, stageC_spectrogram.cpp
// and iq2spectrogram.cpp, which previously each carried a copy.
#pragma once
#include <sycl/sycl.hpp>
#include <cufft.h>
#include <cstdio>
#include <exception>

#define CUFFT_CHECK(x) do { cufftResult r=(x); \
  if(r!=CUFFT_SUCCESS){fprintf(stderr,"cuFFT error %d at %s:%d\n",r,__FILE__,__LINE__);return 1;} }while(0)

namespace cufft_util {

// In-order queue that logs (rather than silently drops) asynchronous SYCL
// exceptions. All the cuFFT-interop programs use this shape of queue.
inline sycl::queue make_inorder_queue() {
  return sycl::queue{
      [](sycl::exception_list exceptions) {
        for (const std::exception_ptr& e : exceptions) {
          try {
            std::rethrow_exception(e);
          } catch (const sycl::exception& ex) {
            std::fprintf(stderr, "asynchronous SYCL exception: %s\n", ex.what());
          }
        }
      },
      sycl::property::queue::in_order{}};
}

// Print the device and verify the queue is CUDA-backed (cuFFT interop only
// works there). Returns false, with a message on stderr, otherwise.
inline bool require_cuda_backend(const sycl::queue& q) {
  std::printf("Device: %s\n",
              q.get_device().get_info<sycl::info::device::name>().c_str());
  if (q.get_device().get_backend() != sycl::backend::cuda) {
    std::fprintf(stderr, "cuFFT interop requires a SYCL queue backed by the CUDA backend\n");
    return false;
  }
  return true;
}

// Enqueue an in-place forward C2C execution of `plan` on `data`, on q's
// native CUDA stream. The result is written through `status` when the
// operation actually runs, so *status must outlive the operation and must
// only be read after q.wait(). Failures do not abort the queue; the caller
// checks *status after waiting and bails out then.
inline void enqueue_exec_c2c_forward(sycl::queue& q, cufftHandle plan,
                                     cufftComplex* data, cufftResult* status) {
  q.submit([&](sycl::handler& cgh) {
    cgh.AdaptiveCpp_enqueue_custom_operation([=](sycl::interop_handle& ih) {
      auto stream = ih.get_native_queue<sycl::backend::cuda>();
      cufftResult r = cufftSetStream(plan, stream);
      if (r == CUFFT_SUCCESS) r = cufftExecC2C(plan, data, data, CUFFT_FORWARD);
      *status = r;
    });
  });
}

} // namespace cufft_util
