// Copyright 2017, 2019 ETH Zürich, Thomas Schöps
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.


#pragma once

#include <cuda_runtime.h>
#include "libvis/logging.h"

#define CUDA_CHECKED_CALL(cuda_call)                                \
  do {                                                              \
    cudaError error = (cuda_call);                                  \
    if (cudaSuccess != error) {                                     \
      LOG(FATAL) << "Cuda Error: " << cudaGetErrorString(error);    \
    }                                                               \
  } while(false)

#define CHECK_CUDA_NO_ERROR()                                       \
  do {                                                              \
    cudaError error = cudaGetLastError();                           \
    if (cudaSuccess != error) {                                     \
      LOG(FATAL) << "Cuda Error: " << cudaGetErrorString(error);    \
    }                                                               \
  } while(false)

// Helper for compiling different versions of bool-templated (kernel) functions
// without duplicating the function calls.
#define COMPILE_OPTION(option, ...) \
  do { \
    if (option) { \
      constexpr bool _##option = true; \
      (void) _##option; \
      __VA_ARGS__; \
    } else { \
      constexpr bool _##option = false; \
      (void) _##option; \
      __VA_ARGS__; \
    } \
  } while (false)

#define COMPILE_OPTION_2(option_a, option_b, ...) \
  do { \
    COMPILE_OPTION(option_a, COMPILE_OPTION(option_b, __VA_ARGS__););\
  } while (false)

#define COMPILE_OPTION_3(option_a, option_b, option_c, ...) \
  do { \
    COMPILE_OPTION(option_a, COMPILE_OPTION(option_b, COMPILE_OPTION(option_c, __VA_ARGS__);););\
  } while (false)

#define COMPILE_OPTION_4(option_a, option_b, option_c, option_d, ...) \
  do { \
    COMPILE_OPTION(option_a, COMPILE_OPTION(option_b, COMPILE_OPTION(option_c, COMPILE_OPTION(option_d, __VA_ARGS__););););\
  } while (false)

#define COMPILE_INT_4_OPTIONS(option, value1, value2, value3, value4, ...) \
  do { \
    if ((option) == (value1)) { \
      constexpr int _##option = (value1); \
      (void) _##option; \
      __VA_ARGS__; \
    } else if ((option) == (value2)) { \
      constexpr int _##option = (value2); \
      (void) _##option; \
      __VA_ARGS__; \
    } else if ((option) == (value3)) { \
      constexpr int _##option = (value3); \
      (void) _##option; \
      __VA_ARGS__; \
    } else if ((option) == (value4)) { \
      constexpr int _##option = (value4); \
      (void) _##option; \
      __VA_ARGS__; \
    } else {LOG(FATAL) << "COMPILE_INT_4_OPTIONS(): Called with unsupported option value.";} \
  } while (false)


namespace vis {

// Returns the required number of CUDA blocks to cover a given domain size,
// given a specific block size.
__forceinline__ __host__ __device__ int GetBlockCount(int domain_size, int block_size) {
  return ((domain_size - 1) / block_size) + 1;
}

}  // namespace vis
