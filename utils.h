#pragma once

// ReSharper disable CppInconsistentNaming
#ifndef DEVICE_RESET
#define DEVICE_RESET cudaDeviceReset();
#endif

#ifndef checkCudaErrors
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

template <typename T>
// ReSharper disable CppInconsistentNaming
void check(T result, char const *const func, const char *const file, int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    DEVICE_RESET
    // Make sure we call CUDA Device Reset before exiting
    exit(EXIT_FAILURE);
  }
}

static const char *_cudaGetErrorEnum(const cudaError_t error) {
  return cudaGetErrorName(error);
}
#endif
// ReSharper restore CppInconsistentNaming