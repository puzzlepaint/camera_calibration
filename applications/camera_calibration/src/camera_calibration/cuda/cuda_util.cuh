// Copyright 2019 ETH Zürich, Thomas Schöps
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
#include <libvis/libvis.h>

namespace vis {

// Computes the L2-norm of a 3-vector.
__forceinline__ __device__ float Norm(const float3& vec) {
#ifdef __CUDA_ARCH__
  return sqrtf(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
#else
  (void) vec;
  return -1;  // We can neither use std::sqrtf nor ::sqrtf here. Either nvcc or the host compiler will complain.
#endif
}

// template <typename Scalar>
// __forceinline__ __device__ Scalar ComputeHuberCost(Scalar residual, Scalar huber_parameter) {
//   const Scalar abs_residual = fabs(residual);
//   if (abs_residual < huber_parameter) {
//     return static_cast<Scalar>(0.5) * residual * residual;
//   } else {
//     return huber_parameter * (abs_residual - static_cast<Scalar>(0.5) * huber_parameter);
//   }
// }

template <typename Scalar>
__forceinline__ __device__ Scalar ComputeHuberCost(Scalar residual_x, Scalar residual_y, Scalar huber_parameter) {
  Scalar squared_residual = residual_x * residual_x + residual_y * residual_y;
  if (squared_residual < huber_parameter * huber_parameter) {
    return static_cast<Scalar>(0.5) * squared_residual;
  } else {
    return huber_parameter * (sqrtf(squared_residual) - static_cast<Scalar>(0.5) * huber_parameter);
  }
}

// template <typename Scalar>
// __forceinline__ __device__ Scalar ComputeHuberWeight(Scalar residual, Scalar huber_parameter) {
//   const Scalar abs_residual = fabs(residual);
//   return (abs_residual < huber_parameter) ? 1 : (huber_parameter / abs_residual);
// }

template <typename Scalar>
__forceinline__ __device__ Scalar ComputeHuberWeight(Scalar residual_x, Scalar residual_y, Scalar huber_parameter) {
  Scalar squared_residual = residual_x * residual_x + residual_y * residual_y;
  return (squared_residual < huber_parameter * huber_parameter) ? 1 : (huber_parameter / sqrtf(squared_residual));
}

// Derived in derive_jacobians.py
template <typename T>
__forceinline__ __device__ void QuaternionJacobianWrtLocalUpdate(
    T w, T x, T y, T z,
    CUDAMatrix<T, 4, 3>* jacobian) {
  T* p = jacobian->data();
  
  // Row 0
  *(p++) = -x;
  *(p++) = -y;
  *(p++) = -z;
  
  // Row 1
  *(p++) = w;
  *(p++) = z;
  *(p++) = -y;
  
  // Row 2
  *(p++) = -z;
  *(p++) = w;
  *(p++) = x;
  
  // Row 3
  *(p++) = y;
  *(p++) = -x;
  *(p++) = w;
}

// Implementation of CUDA's "atomicAdd()" that works for both floats and doubles
// (i.e., can be used with PCGScalar).
template<typename T> __forceinline__ __device__ T atomicAddFloatOrDouble(T* address, T value);

template<> __forceinline__ __device__ float atomicAddFloatOrDouble(float* address, float value) {
  return atomicAdd(address, value);
}

// Implementation of CUDA's "atomicAdd()" for doubles. Directly taken from the
// CUDA C Programming Guide.
template<> __forceinline__ __device__ double atomicAddFloatOrDouble(double* address, double value) {
  unsigned long long int* address_as_ull =
                            (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(value + __longlong_as_double(assumed)));

  // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}

// Efficient sum over all threads in a CUDA thread block.
// Every thread participating here must aim to use the same dest (since only the
// value of thread 0 will be used), and all threads in the block must participate.
// For varying dest, use atomicAddFloatOrDouble().
template<int block_width, int block_height>
__forceinline__ __device__ void BlockedAtomicSum(
    PCGScalar* dest,
    PCGScalar value,
    bool valid,
    typename cub::BlockReduce<PCGScalar, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height>::TempStorage* storage) {
  typedef cub::BlockReduce<PCGScalar, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceScalar;
  const PCGScalar sum = BlockReduceScalar(*storage).Sum(valid ? value : 0.f);
  if (threadIdx.x == 0 && (block_height == 1 || threadIdx.y == 0)) {
    atomicAddFloatOrDouble(dest, sum);
  }
}

}
