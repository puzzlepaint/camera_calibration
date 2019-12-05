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

#include "libvis/camera.h"
#include "libvis/cuda/cuda_buffer.h"
#include "libvis/eigen.h"
#include "libvis/libvis.h"

namespace vis {

class CUDAUnprojectionLookup2D_;

// Creates a lookup texture for 2D unprojection of image pixels to directions,
// i.e., assuming that the z component of the unprojected vectors is always 1.
class CUDAUnprojectionLookup2D {
 public:
  inline CUDAUnprojectionLookup2D(const Image<float2>& lookup_buffer_cpu, cudaStream_t stream)
      : lookup_buffer_(lookup_buffer_cpu.height(), lookup_buffer_cpu.width()) {
    Initialize(lookup_buffer_cpu, stream);
  }
  
  inline CUDAUnprojectionLookup2D(const Camera& camera, cudaStream_t stream)
      : lookup_buffer_(camera.height(), camera.width()) {
    IDENTIFY_CAMERA(camera, Initialize(_camera, stream));
  }
  
  inline ~CUDAUnprojectionLookup2D() {
    cudaDestroyTextureObject(lookup_texture_);
  }
  
  inline cudaTextureObject_t lookup_texture() const {
    return lookup_texture_;
  }
  
 private:
  template <typename CameraT>
  void Initialize(const CameraT& camera, cudaStream_t stream) {
    Image<float2> lookup_buffer_cpu(camera.width(), camera.height());
    for (int y = 0; y < camera.height(); ++ y) {
      for (int x = 0; x < camera.width(); ++ x) {
        Vec2f dir = camera.UnprojectFromPixelCenterConv(Vec2d(x, y).cast<typename CameraT::ScalarT>()).template cast<float>().template topRows<2>();
        lookup_buffer_cpu(x, y) = make_float2(dir.x(), dir.y());
      }
    }
    
    Initialize(lookup_buffer_cpu, stream);
  }
  
  void Initialize(const Image<float2>& lookup_buffer_cpu, cudaStream_t stream) {
    lookup_buffer_.UploadAsync(stream, lookup_buffer_cpu);
    lookup_buffer_.CreateTextureObject(
        cudaAddressModeClamp, cudaAddressModeClamp,
        cudaFilterModeLinear, cudaReadModeElementType,
        false, &lookup_texture_);
    cudaStreamSynchronize(stream);
  }
  
  CUDABuffer<float2> lookup_buffer_;
  cudaTextureObject_t lookup_texture_;
};

}
