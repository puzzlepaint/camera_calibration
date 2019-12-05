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

#include "libvis/cuda/patch_match_stereo.cuh"

#include <math_constants.h>

#include "libvis/cuda/cuda_auto_tuner.h"
#include "libvis/cuda/cuda_unprojection_lookup.cuh"
#include "libvis/cuda/cuda_util.cuh"
#include "libvis/cuda/cuda_util.h"
#include "libvis/cuda/patch_match_stereo_cost.cuh"
#include "libvis/cuda/patch_match_stereo_util.cuh"

namespace vis {

__global__ void PatchMatchPropagationStepCUDAKernel(
    StereoParametersSingleCUDA p,
    int match_metric,
    float second_best_min_distance_factor,
    CUDABuffer_<float> best_inv_depth_map) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (x >= p.context_radius &&
      y >= p.context_radius &&
      x < p.inv_depth_map.width() - p.context_radius &&
      y < p.inv_depth_map.height() - p.context_radius) {
    // "Pulling" the values inwards.
    float2 nxy = p.reference_unprojection_lookup.UnprojectPoint(x, y);
    
    #pragma unroll
    for (int dy = -1; dy <= 1; ++ dy) {
      #pragma unroll
      for (int dx = -1; dx <= 1; ++ dx) {
        if ((dx == 0 && dy == 0) ||
            (dx != 0 && dy != 0)) {
          continue;
        }
        
        // Compute inv_depth for propagating the pixel at (x + dx, y + dy) to the center pixel.
        float2 other_nxy = p.reference_unprojection_lookup.UnprojectPoint(x + dx, y + dy);
        
        float other_inv_depth = p.inv_depth_map(y + dy, x + dx);
        float other_depth = 1.f / other_inv_depth;
        
        char2 other_normal_xy_char = p.normals(y + dy, x + dx);
        const float2 other_normal_xy = make_float2(
            other_normal_xy_char.x * (1 / 127.f), other_normal_xy_char.y * (1 / 127.f));
        float other_normal_z = -sqrtf(1.f - other_normal_xy.x * other_normal_xy.x - other_normal_xy.y * other_normal_xy.y);
        
        float plane_d = (other_nxy.x * other_depth) * other_normal_xy.x + (other_nxy.y * other_depth) * other_normal_xy.y + other_depth * other_normal_z;
        
        float inv_depth = CalculatePlaneInvDepth2(plane_d, other_normal_xy, other_normal_z, nxy.x, nxy.y);
        
        // Test whether to propagate
        float proposal_costs = ComputeCosts(
            x, y,
            other_normal_xy,
            inv_depth,
            p,
            match_metric,
            second_best_min_distance_factor,
            best_inv_depth_map);
        
        if (!::isnan(proposal_costs) && !(proposal_costs >= p.costs(y, x))) {
          p.costs(y, x) = proposal_costs;
          
          // NOTE: Other threads could read these values while they are written,
          //       but it should not be very severe if that happens.
          //       Could use ping-pong buffers to avoid that.
          p.normals(y, x) = make_char2(other_normal_xy.x * 127.f, other_normal_xy.y * 127.f);
          p.inv_depth_map(y, x) = inv_depth;
        }
      }  // loop over dx
    }  // loop over dy
  }
}

void PatchMatchPropagationStepCUDA(
    const StereoParametersSingle& p,
    int match_metric,
    float second_best_min_distance_factor,
    CUDABuffer_<float>* best_inv_depth_map) {
  CHECK_CUDA_NO_ERROR();
  CUDA_AUTO_TUNE_2D(
      PatchMatchPropagationStepCUDAKernel,
      16, 16,
      p.inv_depth_map.width(), p.inv_depth_map.height(),
      0, p.stream,
      /* kernel parameters */
      StereoParametersSingleCUDA(p),
      match_metric,
      second_best_min_distance_factor,
      best_inv_depth_map ? *best_inv_depth_map : CUDABuffer_<float>());
  CHECK_CUDA_NO_ERROR();
}


__global__ void PatchMatchPropagationStepCUDAKernel(
    StereoParametersMultiCUDA p,
    int match_metric,
    float second_best_min_distance_factor,
    CUDABuffer_<float> best_inv_depth_map) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (x >= p.context_radius &&
      y >= p.context_radius &&
      x < p.inv_depth_map.width() - p.context_radius &&
      y < p.inv_depth_map.height() - p.context_radius) {
    // "Pulling" the values inwards.
    float2 nxy = p.reference_unprojection_lookup.UnprojectPoint(x, y);
    
    #pragma unroll
    for (int dy = -1; dy <= 1; ++ dy) {
      #pragma unroll
      for (int dx = -1; dx <= 1; ++ dx) {
        if ((dx == 0 && dy == 0) ||
            (dx != 0 && dy != 0)) {
          continue;
        }
        
        // Compute inv_depth for propagating the pixel at (x + dx, y + dy) to the center pixel.
        float2 other_nxy = p.reference_unprojection_lookup.UnprojectPoint(x + dx, y + dy);
        
        float other_inv_depth = p.inv_depth_map(y + dy, x + dx);
        float other_depth = 1.f / other_inv_depth;
        
        char2 other_normal_xy_char = p.normals(y + dy, x + dx);
        const float2 other_normal_xy = make_float2(
            other_normal_xy_char.x * (1 / 127.f), other_normal_xy_char.y * (1 / 127.f));
        float other_normal_z = -sqrtf(1.f - other_normal_xy.x * other_normal_xy.x - other_normal_xy.y * other_normal_xy.y);
        
        float plane_d = (other_nxy.x * other_depth) * other_normal_xy.x + (other_nxy.y * other_depth) * other_normal_xy.y + other_depth * other_normal_z;
        
        float inv_depth = CalculatePlaneInvDepth2(plane_d, other_normal_xy, other_normal_z, nxy.x, nxy.y);
        
        // Test whether to propagate
        const char2 normal_char = p.normals(y, x);
        if (IsCostOfProposedChangeLower(
            x, y,
            make_float2(normal_char.x * (1 / 127.f), normal_char.y * (1 / 127.f)),
            p.inv_depth_map(y, x),
            other_normal_xy,
            inv_depth,
            p,
            match_metric,
            second_best_min_distance_factor,
            best_inv_depth_map)) {
          // NOTE: Other threads could read these values while they are written,
          //       but it should not be very severe if that happens.
          //       Could use ping-pong buffers to avoid that.
          p.normals(y, x) = make_char2(other_normal_xy.x * 127.f, other_normal_xy.y * 127.f);
          p.inv_depth_map(y, x) = inv_depth;
        }
      }  // loop over dx
    }  // loop over dy
  }  // check for the thread's validity
}

void PatchMatchPropagationStepCUDA(
    const StereoParametersMulti& p,
    int match_metric,
    float second_best_min_distance_factor,
    CUDABuffer_<float>* best_inv_depth_map) {
  CHECK_CUDA_NO_ERROR();
  CUDA_AUTO_TUNE_2D(
      PatchMatchPropagationStepCUDAKernel,
      16, 16,
      p.inv_depth_map.width(), p.inv_depth_map.height(),
      0, p.stream,
      /* kernel parameters */
      StereoParametersMultiCUDA(p),
      match_metric,
      second_best_min_distance_factor,
      best_inv_depth_map ? *best_inv_depth_map : CUDABuffer_<float>());
  CHECK_CUDA_NO_ERROR();
}

}
