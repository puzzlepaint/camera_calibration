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

#define RUN_WITH_THREE_OPTION_PAIRS( \
    option_a, option_b, \
    value_a_1, value_b_1, \
    value_a_2, value_b_2, \
    value_a_3, value_b_3, ...) \
  do { \
    { \
      constexpr bool _##option_a = (value_a_1); \
      constexpr bool _##option_b = (value_b_1); \
      __VA_ARGS__; \
    } \
    { \
      constexpr bool _##option_a = (value_a_2); \
      constexpr bool _##option_b = (value_b_2); \
      __VA_ARGS__; \
    } \
    { \
      constexpr bool _##option_a = (value_a_3); \
      constexpr bool _##option_b = (value_b_3); \
      __VA_ARGS__; \
    } \
  } while (false)


template <bool mutate_depth, bool mutate_normal>
__global__ void PatchMatchMutationStepCUDAKernel(
    StereoParametersSingleCUDA p,
    int match_metric,
    float max_normal_2d_length,
    float step_range,
    CUDABuffer_<curandState> random_states,
    float second_best_min_distance_factor,
    CUDABuffer_<float> best_inv_depth_map) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (x >= p.context_radius &&
      y >= p.context_radius &&
      x < p.inv_depth_map.width() - p.context_radius &&
      y < p.inv_depth_map.height() - p.context_radius) {
    // Get current depth
    float proposed_inv_depth = p.inv_depth_map(y, x);
    
    // Mutate depth?
    if (mutate_depth) {
      proposed_inv_depth =
          max(kMinInvDepth,
              fabsf(proposed_inv_depth + step_range *
                  (curand_uniform(&random_states(y, x)) - 0.5f)));
    }
    
    // Get current normal
    constexpr float kRandomNormalRange = 1.0f;
    const char2 proposed_normal_char = p.normals(y, x);
    float2 proposed_normal = make_float2(
        proposed_normal_char.x * (1 / 127.f), proposed_normal_char.y * (1 / 127.f));
    
    // Mutate normal?
    if (mutate_normal) {
      proposed_normal.x += kRandomNormalRange * (curand_uniform(&random_states(y, x)) - 0.5f);
      proposed_normal.y += kRandomNormalRange * (curand_uniform(&random_states(y, x)) - 0.5f);
      float length = sqrtf(proposed_normal.x * proposed_normal.x + proposed_normal.y * proposed_normal.y);
      if (length > max_normal_2d_length) {
        proposed_normal.x *= max_normal_2d_length / length;
        proposed_normal.y *= max_normal_2d_length / length;
      }
    }
    
    // Test whether to accept the proposal
    float proposal_costs = ComputeCosts(
        x, y,
        proposed_normal,
        proposed_inv_depth,
        p,
        match_metric,
        second_best_min_distance_factor,
        best_inv_depth_map);
    
    if (!::isnan(proposal_costs) && !(proposal_costs >= p.costs(y, x))) {
      p.costs(y, x) = proposal_costs;
      p.normals(y, x) = make_char2(proposed_normal.x * 127.f, proposed_normal.y * 127.f);
      p.inv_depth_map(y, x) = proposed_inv_depth;
    }
  }
}

void PatchMatchMutationStepCUDA(
    const StereoParametersSingle& p,
    int match_metric,
    float max_normal_2d_length,
    float step_range,
    CUDABuffer_<curandState>* random_states,
    float second_best_min_distance_factor,
    CUDABuffer_<float>* best_inv_depth_map) {
  CHECK_CUDA_NO_ERROR();
  RUN_WITH_THREE_OPTION_PAIRS(
      mutate_depth, mutate_normal,
      true, true,   //   mutate_depth &&   mutate_normal
      true, false,  //   mutate_depth && ! mutate_normal
      false, true,  // ! mutate_depth &&   mutate_normal
      CUDA_AUTO_TUNE_2D_TEMPLATED(
      PatchMatchMutationStepCUDAKernel,
      16, 16,
      p.inv_depth_map.width(), p.inv_depth_map.height(),
      0, p.stream,
      TEMPLATE_ARGUMENTS(_mutate_depth, _mutate_normal),
      /* kernel parameters */
      StereoParametersSingleCUDA(p),
      match_metric,
      max_normal_2d_length,
      step_range,
      *random_states,
      second_best_min_distance_factor,
      best_inv_depth_map ? *best_inv_depth_map : CUDABuffer_<float>()));
  CHECK_CUDA_NO_ERROR();
}


template <bool mutate_depth, bool mutate_normal>
__global__ void PatchMatchMutationStepCUDAKernel(
    StereoParametersMultiCUDA p,
    int match_metric,
    float max_normal_2d_length,
    float step_range,
    CUDABuffer_<curandState> random_states,
    float second_best_min_distance_factor,
    CUDABuffer_<float> best_inv_depth_map) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (x >= p.context_radius &&
      y >= p.context_radius &&
      x < p.inv_depth_map.width() - p.context_radius &&
      y < p.inv_depth_map.height() - p.context_radius) {
    // Get current depth
    float proposed_inv_depth = p.inv_depth_map(y, x);
    
    // Mutate depth?
    if (mutate_depth) {
      proposed_inv_depth =
          max(kMinInvDepth,
              fabsf(proposed_inv_depth + step_range *
                  (curand_uniform(&random_states(y, x)) - 0.5f)));
    }
    
    // Get current normal
    constexpr float kRandomNormalRange = 1.0f;
    const char2 normal_char = p.normals(y, x);
    float2 proposed_normal = make_float2(
        normal_char.x * (1 / 127.f), normal_char.y * (1 / 127.f));
    
    // Mutate normal?
    if (mutate_normal) {
      proposed_normal.x += kRandomNormalRange * (curand_uniform(&random_states(y, x)) - 0.5f);
      proposed_normal.y += kRandomNormalRange * (curand_uniform(&random_states(y, x)) - 0.5f);
      float length = sqrtf(proposed_normal.x * proposed_normal.x + proposed_normal.y * proposed_normal.y);
      if (length > max_normal_2d_length) {
        proposed_normal.x *= max_normal_2d_length / length;
        proposed_normal.y *= max_normal_2d_length / length;
      }
    }
    
    // Test whether to accept the proposal
    if (IsCostOfProposedChangeLower(
        x, y,
        make_float2(normal_char.x * (1 / 127.f), normal_char.y * (1 / 127.f)),
        p.inv_depth_map(y, x),
        proposed_normal,
        proposed_inv_depth,
        p,
        match_metric,
        second_best_min_distance_factor,
        best_inv_depth_map)) {
      p.normals(y, x) = make_char2(proposed_normal.x * 127.f, proposed_normal.y * 127.f);
      p.inv_depth_map(y, x) = proposed_inv_depth;
    }
  }
}

void PatchMatchMutationStepCUDA(
    const StereoParametersMulti& p,
    int match_metric,
    float max_normal_2d_length,
    float step_range,
    CUDABuffer_<curandState>* random_states,
    float second_best_min_distance_factor,
    CUDABuffer_<float>* best_inv_depth_map) {
  CHECK_CUDA_NO_ERROR();
  RUN_WITH_THREE_OPTION_PAIRS(
      mutate_depth, mutate_normal,
      true, true,   //   mutate_depth &&   mutate_normal
      true, false,  //   mutate_depth && ! mutate_normal
      false, true,  // ! mutate_depth &&   mutate_normal
      CUDA_AUTO_TUNE_2D_TEMPLATED(
      PatchMatchMutationStepCUDAKernel,
      16, 16,
      p.inv_depth_map.width(), p.inv_depth_map.height(),
      0, p.stream,
      TEMPLATE_ARGUMENTS(_mutate_depth, _mutate_normal),
      /* kernel parameters */
      StereoParametersMultiCUDA(p),
      match_metric,
      max_normal_2d_length,
      step_range,
      *random_states,
      second_best_min_distance_factor,
      best_inv_depth_map ? *best_inv_depth_map : CUDABuffer_<float>()));
  CHECK_CUDA_NO_ERROR();
}

}
