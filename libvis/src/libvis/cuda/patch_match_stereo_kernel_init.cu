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

__global__ void InitPatchMatchCUDAKernel(
    StereoParametersSingleCUDA p,
    int match_metric,
    float max_normal_2d_length,
    float inv_min_depth,
    float inv_max_depth,
    CUDABuffer_<curandState> random_states,
    float second_best_min_distance_factor,
    CUDABuffer_<float> best_inv_depth_map) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (x >= p.context_radius &&
      y >= p.context_radius &&
      x < p.inv_depth_map.width() - p.context_radius &&
      y < p.inv_depth_map.height() - p.context_radius) {
    // Initialize random states
    // TODO: Would it be better to do this only once for each PatchMatchStereo object?
    int id = x + p.inv_depth_map.width() * y;
    curand_init(id, 0, 0, &random_states(y, x));
    
    // Initialize random initial normals
    constexpr float kNormalRange = 1.0f;
    float2 normal_xy;
    normal_xy.x = kNormalRange * (curand_uniform(&random_states(y, x)) - 0.5f);
    normal_xy.y = kNormalRange * (curand_uniform(&random_states(y, x)) - 0.5f);
    float length = sqrtf(normal_xy.x * normal_xy.x + normal_xy.y * normal_xy.y);
    if (length > max_normal_2d_length) {
      normal_xy.x *= max_normal_2d_length / length;
      normal_xy.y *= max_normal_2d_length / length;
    }
    p.normals(y, x) = make_char2(normal_xy.x * 127.f, normal_xy.y * 127.f);
    
    // Initialize random initial depths
    const float inv_depth = inv_max_depth + (inv_min_depth - inv_max_depth) * curand_uniform(&random_states(y, x));
    p.inv_depth_map(y, x) = inv_depth;
    
    // Compute initial costs
    p.costs(y, x) = ComputeCosts(
        x, y,
        normal_xy,
        inv_depth,
        p,
        match_metric,
        second_best_min_distance_factor,
        best_inv_depth_map);
  }
}

void InitPatchMatchCUDA(
    const StereoParametersSingle& p,
    int match_metric,
    float max_normal_2d_length,
    float inv_min_depth,
    float inv_max_depth,
    CUDABuffer_<curandState>* random_states,
    float second_best_min_distance_factor,
    CUDABuffer_<float>* best_inv_depth_map) {
  InitPatchMatchSamples();  // TODO: Do this separately
  
  CHECK_CUDA_NO_ERROR();
  CUDA_AUTO_TUNE_2D(
      InitPatchMatchCUDAKernel,
      16, 16,
      p.inv_depth_map.width(), p.inv_depth_map.height(),
      0, p.stream,
      /* kernel parameters */
      StereoParametersSingleCUDA(p),
      match_metric,
      max_normal_2d_length,
      inv_min_depth,
      inv_max_depth,
      *random_states,
      second_best_min_distance_factor,
      best_inv_depth_map ? *best_inv_depth_map : CUDABuffer_<float>());
  CHECK_CUDA_NO_ERROR();
}


__global__ void InitPatchMatchCUDAKernel(
    StereoParametersMultiCUDA p,
    float max_normal_2d_length,
    float inv_min_depth,
    float inv_max_depth,
    CUDABuffer_<curandState> random_states) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (x >= p.context_radius &&
      y >= p.context_radius &&
      x < p.inv_depth_map.width() - p.context_radius &&
      y < p.inv_depth_map.height() - p.context_radius) {
    // Initialize random states
    // TODO: Would it be better to do this only once for each PatchMatchStereo object?
    int id = x + p.inv_depth_map.width() * y;
    curand_init(id, 0, 0, &random_states(y, x));
    
    // Initialize random initial normals
    constexpr float kNormalRange = 1.0f;
    float2 normal_xy;
    normal_xy.x = kNormalRange * (curand_uniform(&random_states(y, x)) - 0.5f);
    normal_xy.y = kNormalRange * (curand_uniform(&random_states(y, x)) - 0.5f);
    float length = sqrtf(normal_xy.x * normal_xy.x + normal_xy.y * normal_xy.y);
    if (length > max_normal_2d_length) {
      normal_xy.x *= max_normal_2d_length / length;
      normal_xy.y *= max_normal_2d_length / length;
    }
    p.normals(y, x) = make_char2(normal_xy.x * 127.f, normal_xy.y * 127.f);
    
    // Initialize random initial depths
    const float inv_depth = inv_max_depth + (inv_min_depth - inv_max_depth) * curand_uniform(&random_states(y, x));
    p.inv_depth_map(y, x) = inv_depth;
  }
}

void InitPatchMatchCUDA(
    const StereoParametersMulti& p,
    float max_normal_2d_length,
    float inv_min_depth,
    float inv_max_depth,
    CUDABuffer_<curandState>* random_states) {
  InitPatchMatchSamples();  // TODO: Do this separately
  
  CHECK_CUDA_NO_ERROR();
  CUDA_AUTO_TUNE_2D(
      InitPatchMatchCUDAKernel,
      16, 16,
      p.inv_depth_map.width(), p.inv_depth_map.height(),
      0, p.stream,
      /* kernel parameters */
      StereoParametersMultiCUDA(p),
      max_normal_2d_length,
      inv_min_depth,
      inv_max_depth,
      *random_states);
  CHECK_CUDA_NO_ERROR();
}

}
