/**
 * @file cufft_interop.hpp
 * @brief Shared plumbing for driving cuFFT from SYCL (AdaptiveCpp) code.
 *
 * The in-order queue with async-exception logging, the CUDA-backend check,
 * and the interop submission that runs a batched in-place C2C FFT on the
 * queue's native CUDA stream. Extracted from stageB_cufft.cpp,
 * stageC_spectrogram.cpp, and iq2spectrogram.cpp, which previously each
 * carried a copy.
 */
#pragma once
#include <sycl/sycl.hpp>
#include <cufft.h>
#include <cstdio>
#include <exception>

/// Run a cuFFT call, printing a message and returning 1 (intended for use
/// directly in `main()`) if it didn't succeed.
/// @param x A `cufftResult`-returning expression.
#define CUFFT_CHECK(x) do { cufftResult r=(x); \
  if(r!=CUFFT_SUCCESS){fprintf(stderr,"cuFFT error %d at %s:%d\n",r,__FILE__,__LINE__);return 1;} }while(0)

/// @namespace cufft_util
/// @brief Shared cuFFT-via-SYCL-interop plumbing used by every
/// cuFFT-calling program (stageB_cufft.cpp, stageC_spectrogram.cpp,
/// iq2spectrogram.cpp).
namespace cufft_util {

/// Construct the in-order queue every cuFFT-interop program uses: it logs
/// (rather than silently drops) asynchronous SYCL exceptions.
/// @return A new in-order queue with an async exception handler installed.
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

/// Print the device name and verify the queue is CUDA-backed (cuFFT
/// interop only works there).
/// @param q Queue to check.
/// @return True if `q`'s device uses the CUDA backend; false (with a
/// message on stderr) otherwise.
inline bool require_cuda_backend(const sycl::queue& q) {
  std::printf("Device: %s\n",
              q.get_device().get_info<sycl::info::device::name>().c_str());
  if (q.get_device().get_backend() != sycl::backend::cuda) {
    std::fprintf(stderr, "cuFFT interop requires a SYCL queue backed by the CUDA backend\n");
    return false;
  }
  return true;
}

/// Enqueue an in-place forward C2C execution of `plan` on `data`, on `q`'s
/// native CUDA stream.
/// @param q Queue to submit on (must be CUDA-backed; see
/// require_cuda_backend()).
/// @param plan A cuFFT plan already created for `data`'s batch size/length.
/// @param data Device buffer to transform in-place.
/// @param[out] status Written with the cuFFT result when the operation
/// actually runs on-device. Must outlive the operation and must only be
/// read after `q.wait()`. Failures do not abort the queue; the caller
/// checks `*status` after waiting and bails out then.
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
