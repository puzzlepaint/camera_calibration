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

__global__ void PatchMatchDiscreteRefinementStepCUDAKernel(
    StereoParametersSingleCUDA p,
    int match_metric,
    int num_steps,
    float range_factor) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (x >= p.context_radius &&
      y >= p.context_radius &&
      x < p.inv_depth_map.width() - p.context_radius &&
      y < p.inv_depth_map.height() - p.context_radius) {
    float original_inv_depth = p.inv_depth_map(y, x);
    
    const char2 normal_char = p.normals(y, x);
    float2 normal = make_float2(
        normal_char.x * (1 / 127.f), normal_char.y * (1 / 127.f));
    
    for (int step = 0; step < num_steps; ++ step) {
      float proposed_inv_depth = (1 + range_factor * 2 * ((step / (num_steps - 1.f)) - 0.5f)) * original_inv_depth;
      
      // Test whether to accept the proposal
      float proposal_costs = ComputeCosts(
          x, y,
          normal,
          proposed_inv_depth,
          p,
          match_metric,
          0,  // TODO: Update if using this function within the second best cost step
          p.inv_depth_map);  // TODO: Update if using this function within the second best cost step
      
      if (!::isnan(proposal_costs) && !(proposal_costs >= p.costs(y, x))) {
        p.costs(y, x) = proposal_costs;
        p.inv_depth_map(y, x) = proposed_inv_depth;
      }
    }
  }
}

void PatchMatchDiscreteRefinementStepCUDA(
    const StereoParametersSingle& p,
    int match_metric,
    int num_steps,
    float range_factor) {
  CHECK_CUDA_NO_ERROR();
  CUDA_AUTO_TUNE_2D(
      PatchMatchDiscreteRefinementStepCUDAKernel,
      16, 16,
      p.inv_depth_map.width(), p.inv_depth_map.height(),
      0, p.stream,
      /* kernel parameters */
      StereoParametersSingleCUDA(p),
      match_metric,
      num_steps,
      range_factor);
  CHECK_CUDA_NO_ERROR();
}


__global__ void PatchMatchDiscreteRefinementStepCUDAKernel(
    StereoParametersMultiCUDA p,
    int match_metric,
    int num_steps,
    float range_factor) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (x >= p.context_radius &&
      y >= p.context_radius &&
      x < p.inv_depth_map.width() - p.context_radius &&
      y < p.inv_depth_map.height() - p.context_radius) {
    float original_inv_depth = p.inv_depth_map(y, x);
    
    const char2 normal_char = p.normals(y, x);
    float2 normal = make_float2(
        normal_char.x * (1 / 127.f), normal_char.y * (1 / 127.f));
    
    for (int step = 0; step < num_steps; ++ step) {
      float proposed_inv_depth = (1 + range_factor * 2 * ((step / (num_steps - 1.f)) - 0.5f)) * original_inv_depth;
      
      // Test whether to accept the proposal
      if (IsCostOfProposedChangeLower(
          x, y,
          normal,
          p.inv_depth_map(y, x),
          normal,
          proposed_inv_depth,
          p,
          match_metric,
          0,  // TODO: Update if using this function within the second best cost step
          p.inv_depth_map)) {  // TODO: Update if using this function within the second best cost step
        p.inv_depth_map(y, x) = proposed_inv_depth;
      }
    }
  }
}

void PatchMatchDiscreteRefinementStepCUDA(
    const StereoParametersMulti& p,
    int match_metric,
    int num_steps,
    float range_factor) {
  CHECK_CUDA_NO_ERROR();
  CUDA_AUTO_TUNE_2D(
      PatchMatchDiscreteRefinementStepCUDAKernel,
      16, 16,
      p.inv_depth_map.width(), p.inv_depth_map.height(),
      0, p.stream,
      /* kernel parameters */
      StereoParametersMultiCUDA(p),
      match_metric,
      num_steps,
      range_factor);
  CHECK_CUDA_NO_ERROR();
}

}
