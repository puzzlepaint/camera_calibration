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
#include "libvis/cuda/pixel_corner_projector.cuh"
#include "libvis/eigen.h"

namespace vis {

struct PixelCornerProjector {
  PixelCornerProjector(
      Camera::Type camera_type,
      int width,
      int height,
      float fx,
      float fy,
      float cx,
      float cy,
      float omega = 0,
      float k1 = 0,
      float k2 = 0,
      float k3 = 0,
      float k4 = 0,
      float p1 = 0,
      float p2 = 0,
      float sx1 = 0,
      float sy1 = 0) {
    d.type = static_cast<int>(camera_type);
    d.width = width;
    d.height = height;
    d.fx = fx;
    d.fy = fy;
    d.cx = cx;
    d.cy = cy;
    d.omega = omega;
    d.two_tan_omega_half = 2.0f * tan(0.5f * omega);
    d.k1 = k1;
    d.k2 = k2;
    d.k3 = k3;
    d.k4 = k4;
    d.p1 = p1;
    d.p2 = p2;
    d.sx1 = sx1;
    d.sy1 = sy1;
  }
  
  PixelCornerProjector(const Camera& camera) {
    d.type = static_cast<int>(camera.type());
    d.width = camera.width();
    d.height = camera.height();
    
    if (camera.type() == Camera::Type::kPinholeCamera4f) {
      const PinholeCamera4f& pinhole_camera = reinterpret_cast<const PinholeCamera4f&>(camera);
      
      d.fx = pinhole_camera.parameters()[0];
      d.fy = pinhole_camera.parameters()[1];
      d.cx = pinhole_camera.parameters()[2];
      d.cy = pinhole_camera.parameters()[3];
    } else if (camera.type() == Camera::Type::kRadtanCamera8d) {
      const RadtanCamera8d& radtan_camera = reinterpret_cast<const RadtanCamera8d&>(camera);
      
      d.k1 = radtan_camera.parameters()[0];
      d.k2 = radtan_camera.parameters()[1];
      d.p1 = radtan_camera.parameters()[2];
      d.p2 = radtan_camera.parameters()[3];
      
      d.fx = radtan_camera.parameters()[4];
      d.fy = radtan_camera.parameters()[5];
      d.cx = radtan_camera.parameters()[6];
      d.cy = radtan_camera.parameters()[7];
    } else if (camera.type() == Camera::Type::kThinPrismFisheyeCamera12d) {
      const ThinPrismFisheyeCamera12d& thinprism_camera = reinterpret_cast<const ThinPrismFisheyeCamera12d&>(camera);
      
      d.k1 = thinprism_camera.parameters()[0];
      d.k2 = thinprism_camera.parameters()[1];
      d.k3 = thinprism_camera.parameters()[2];
      d.k4 = thinprism_camera.parameters()[3];
      d.p1 = thinprism_camera.parameters()[4];
      d.p2 = thinprism_camera.parameters()[5];
      d.sx1 = thinprism_camera.parameters()[6];
      d.sy1 = thinprism_camera.parameters()[7];
      
      d.fx = thinprism_camera.parameters()[8];
      d.fy = thinprism_camera.parameters()[9];
      d.cx = thinprism_camera.parameters()[10];
      d.cy = thinprism_camera.parameters()[11];
    } else if (camera.type() == Camera::Type::kNonParametricBicubicProjectionCamerad) {
      const NonParametricBicubicProjectionCamerad& np_camera = reinterpret_cast<const NonParametricBicubicProjectionCamerad&>(camera);
      
      d.resolution_x = np_camera.parameters()[0];
      d.resolution_y = np_camera.parameters()[1];
      d.min_nx = np_camera.parameters()[2];
      d.min_ny = np_camera.parameters()[3];
      d.max_nx = np_camera.parameters()[4];
      d.max_ny = np_camera.parameters()[5];
      
      vector<float2> grid_data(d.resolution_x * d.resolution_y);
      for (int i = 0; i < d.resolution_x * d.resolution_y; ++ i) {
        grid_data[i] = make_float2(
            np_camera.parameters()[6 + 2 * i + 0],
            np_camera.parameters()[6 + 2 * i + 1]);
      }
      grid2.reset(new CUDABuffer<float2>(d.resolution_y, d.resolution_x));
      grid2->UploadAsync(/*stream*/ 0, grid_data.data());
      d.grid2 = grid2->ToCUDA();
    } else {
      LOG(FATAL) << "Constructor called with unsupported camera type";
    }
  }
  
  PixelCornerProjector(
      int width, int height,
      int min_x, int min_y,
      int max_x, int max_y,
      const Image<float3>& cpu_grid3) {
    d.type = static_cast<int>(Camera::Type::kInvalid);
    d.width = width;
    d.height = height;
    
    d.min_nx = min_x;
    d.min_ny = min_y;
    d.max_nx = max_x;
    d.max_ny = max_y;
    
    grid3.reset(new CUDABuffer<float3>(cpu_grid3.height(), cpu_grid3.width()));
    grid3->UploadAsync(/*stream*/ 0, cpu_grid3);
    d.grid3 = grid3->ToCUDA();
  }
  
  PixelCornerProjector(const PixelCornerProjector& other) = delete;
  
  __forceinline__ const PixelCornerProjector_& ToCUDA() const {
    return d;
  }
  
 private:
  CUDABufferPtr<float2> grid2;
  CUDABufferPtr<float3> grid3;
  PixelCornerProjector_ d;
};

}
