/**
 * @file sycl_util.hpp
 * @brief Shared SYCL USM allocation helper.
 *
 * `sycl::malloc_device` returns `nullptr` (rather than throwing) when the
 * driver can't satisfy an allocation, so every call site needs the same
 * null check or it silently turns into a null-pointer kernel crash instead
 * of a clear message. Centralized here so it isn't skipped anywhere.
 */
#pragma once
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>

/// @namespace sycl_util
/// @brief Small SYCL-related helpers shared across the device-side programs.
namespace sycl_util {

/// Allocate device USM, exiting with a clear message instead of returning
/// `nullptr` on failure.
/// @tparam T Element type to allocate.
/// @param count Number of elements of type `T` to allocate.
/// @param q Queue whose device the memory is allocated on.
/// @param what Short label identifying the allocation in the error message
/// (e.g. the variable name) if it fails.
/// @return A valid, non-null device pointer. Never returns on failure --
/// prints an error and calls `std::exit(1)` instead.
template <typename T>
inline T* malloc_device_checked(size_t count, sycl::queue& q, const char* what) {
  T* p = sycl::malloc_device<T>(count, q);
  if (!p) {
    std::fprintf(stderr, "%s: sycl::malloc_device failed (%zu bytes)\n",
                 what, count * sizeof(T));
    std::exit(1);
  }
  return p;
}

} // namespace sycl_util
