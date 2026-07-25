// sycl::malloc_device returns nullptr (rather than throwing) when the
// driver can't satisfy an allocation, so every call site needs the same
// null check or it silently turns into a null-pointer kernel crash instead
// of a clear message. Centralized here so it isn't skipped anywhere.
#pragma once
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>

namespace sycl_util {

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
