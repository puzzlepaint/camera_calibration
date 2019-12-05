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

#include "libvis/libvis.h"

namespace vis {

struct CUDAMatrix3x3 {
  __forceinline__ __host__ __device__ CUDAMatrix3x3() {}
  
  template <typename T> __host__ explicit
  CUDAMatrix3x3(const T& matrix) {
    row0.x = matrix(0, 0);
    row0.y = matrix(0, 1);
    row0.z = matrix(0, 2);
    row1.x = matrix(1, 0);
    row1.y = matrix(1, 1);
    row1.z = matrix(1, 2);
    row2.x = matrix(2, 0);
    row2.y = matrix(2, 1);
    row2.z = matrix(2, 2);
  }

  __forceinline__ __host__ __device__
  float3 operator* (const float3& point) const {
    return make_float3(
        row0.x * point.x + row0.y * point.y + row0.z * point.z,
        row1.x * point.x + row1.y * point.y + row1.z * point.z,
        row2.x * point.x + row2.y * point.y + row2.z * point.z);
  }
  
  float3 row0;
  float3 row1;
  float3 row2;
};

struct CUDAMatrix3x4 {
  __forceinline__ __host__ __device__ CUDAMatrix3x4() {}
  
  template <typename T> __host__ explicit
  CUDAMatrix3x4(const T& matrix) {
    row0.x = matrix(0, 0);
    row0.y = matrix(0, 1);
    row0.z = matrix(0, 2);
    row0.w = matrix(0, 3);
    row1.x = matrix(1, 0);
    row1.y = matrix(1, 1);
    row1.z = matrix(1, 2);
    row1.w = matrix(1, 3);
    row2.x = matrix(2, 0);
    row2.y = matrix(2, 1);
    row2.z = matrix(2, 2);
    row2.w = matrix(2, 3);
  }

  __forceinline__ __host__ __device__
  float3 operator* (const float3& point) const {
    return make_float3(
        row0.x * point.x + row0.y * point.y + row0.z * point.z + row0.w,
        row1.x * point.x + row1.y * point.y + row1.z * point.z + row1.w,
        row2.x * point.x + row2.y * point.y + row2.z * point.z + row2.w);
  }
  
  __forceinline__ __host__ __device__
  bool MultiplyIfResultZIsPositive(const float3& point, float3* result) const {
    result->z = row2.x * point.x + row2.y * point.y + row2.z * point.z + row2.w;
    if (result->z <= 0.f) {
      return false;
    }
    result->x = row0.x * point.x + row0.y * point.y + row0.z * point.z + row0.w;
    result->y = row1.x * point.x + row1.y * point.y + row1.z * point.z + row1.w;
    return true;
  }
  
  __forceinline__ __host__ __device__
  float3 Rotate(const float3& point) const {
    return make_float3(
        row0.x * point.x + row0.y * point.y + row0.z * point.z,
        row1.x * point.x + row1.y * point.y + row1.z * point.z,
        row2.x * point.x + row2.y * point.y + row2.z * point.z);
  }
  
  float4 row0;
  float4 row1;
  float4 row2;
};

}
