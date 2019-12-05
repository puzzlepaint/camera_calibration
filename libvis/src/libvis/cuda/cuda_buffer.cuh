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

#include <cstdio>
#include <stdint.h>

#include <cuda_runtime.h>

#include "libvis/libvis.h"

namespace vis {
template<typename T>
class CUDABuffer;

// Part of a CUDABuffer which is usable from CUDA code.
template<typename T>
class CUDABuffer_ {
 public:
  inline CUDABuffer_() {}
  inline CUDABuffer_(T* address, int height, int width, size_t pitch)
      : address_(address),
        height_(height),
        width_(width),
        pitch_(pitch) {}

  void Clear(T value, cudaStream_t stream);
  void SetTo(cudaTextureObject_t texture, cudaStream_t stream);
  void SetToReadModeNormalized(cudaTextureObject_t texture, cudaStream_t stream);
  void SetTo(const CUDABuffer_<T>& other, cudaStream_t stream);

  __forceinline__ __device__ T& operator()(unsigned int y,
                                           unsigned int x) {
#ifdef DEBUG
    if (x >= width_ || y >= height_) {
      printf("Out of bounds buffer access at (%i, %i), buffer size (%i, %i)\n",
             x, y, width_, height_);
    }
#endif
    return *(reinterpret_cast<T*>(reinterpret_cast<int8_t*>(address_)
                                  + y * pitch_) + x);
  }

  __forceinline__ __device__ const T& operator()(unsigned int y,
                                                 unsigned int x) const {
#ifdef DEBUG
    if (x >= width_ || y >= height_) {
      printf("Out of bounds buffer access at (%i, %i), buffer size (%i, %i)\n",
             x, y, width_, height_);
    }
#endif
    return *(reinterpret_cast<const T*>(
        reinterpret_cast<const int8_t*>(address_) + y*pitch_) + x);
  }
  
  __forceinline__ __device__ T& operator()(const int2& pixel) {
    return operator()(pixel.y, pixel.x);
  }
  
    __forceinline__ __device__ const T& operator()(const int2& pixel) const {
    return operator()(pixel.y, pixel.x);
  }
  
  __forceinline__ __device__ T& operator()(const uint2& pixel) {
    return operator()(pixel.y, pixel.x);
  }
  
  __forceinline__ __device__ const T& operator()(const uint2& pixel) const {
    return operator()(pixel.y, pixel.x);
  }

  __forceinline__ __host__ __device__ T* address() const {
    return address_;
  }
  __forceinline__ __host__ __device__ int width() const {
    return width_;
  }
  __forceinline__ __host__ __device__ int height() const {
    return height_;
  }
  __forceinline__ __host__ __device__ size_t pitch() const {
    return pitch_;
  }

 private:
 friend class CUDABuffer<T>;

  T* address_;
  int height_;
  int width_;
  size_t pitch_;
};
}  // namespace vis
