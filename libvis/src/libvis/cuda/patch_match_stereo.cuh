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
#include <curand_kernel.h>

#include "libvis/cuda/cuda_buffer.cuh"
#include "libvis/cuda/cuda_matrix.cuh"
#include "libvis/cuda/pixel_corner_projector.cuh"
#include "libvis/libvis.h"

namespace vis {

constexpr int kPatchMatchStereo_MatchMetric_SSD = 0;
constexpr int kPatchMatchStereo_MatchMetric_ZNCC = 1;
constexpr int kPatchMatchStereo_MatchMetric_Census = 2;

/// Groups common parameters to CUDA kernels for stereo estimation, for a single stereo image.
struct StereoParametersSingle {
  cudaStream_t stream;
  int context_radius;
  cudaTextureObject_t reference_unprojection_lookup;
  CUDABuffer_<u8> reference_image;
  cudaTextureObject_t reference_texture;
  PixelCornerProjector_ stereo_camera;
  CUDABuffer_<u8> mask;
  
  CUDAMatrix3x4 stereo_tr_reference;
  cudaTextureObject_t stereo_image;
  
  CUDABuffer_<float> inv_depth_map;
  CUDABuffer_<char2> normals;
  CUDABuffer_<float> costs;
};

/// Groups common parameters to CUDA kernels for stereo estimation, for multiple stereo images.
struct StereoParametersMulti {
  cudaStream_t stream;
  int context_radius;
  cudaTextureObject_t reference_unprojection_lookup;
  CUDABuffer_<u8> reference_image;
  cudaTextureObject_t reference_texture;
  PixelCornerProjector_ stereo_camera;
  CUDABuffer_<u8> mask;
  
  int num_stereo_images;
  CUDAMatrix3x4* stereo_tr_reference;
  cudaTextureObject_t* stereo_images;
  
  CUDABuffer_<float> inv_depth_map;
  CUDABuffer_<char2> normals;
};


// Image-pair variant
void InitPatchMatchCUDA(
    const StereoParametersSingle& p,
    int match_metric,
    float max_normal_2d_length,
    float inv_min_depth,
    float inv_max_depth,
    CUDABuffer_<curandState>* random_states,
    float second_best_min_distance_factor = 0,
    CUDABuffer_<float>* best_inv_depth_map = nullptr);

// Multiple-image variant
void InitPatchMatchCUDA(
    const StereoParametersMulti& p,
    float max_normal_2d_length,
    float inv_min_depth,
    float inv_max_depth,
    CUDABuffer_<curandState>* random_states);


// Image-pair variant
void PatchMatchMutationStepCUDA(
    const StereoParametersSingle& p,
    int match_metric,
    float max_normal_2d_length,
    float step_range,
    CUDABuffer_<curandState>* random_states,
    float second_best_min_distance_factor = 0,
    CUDABuffer_<float>* best_inv_depth_map = nullptr);

// Multiple-image variant
void PatchMatchMutationStepCUDA(
    const StereoParametersMulti& p,
    int match_metric,
    float max_normal_2d_length,
    float step_range,
    CUDABuffer_<curandState>* random_states,
    float second_best_min_distance_factor = 0,
    CUDABuffer_<float>* best_inv_depth_map = nullptr);


// void PatchMatchOptimizationStepCUDA(
//     const StereoParametersSingle& p,
//     int match_metric,
//     float max_normal_2d_length,
//     CUDABuffer_<curandState>* random_states,
//     CUDABuffer_<float>* lambda);


// Image-pair variant
void PatchMatchPropagationStepCUDA(
    const StereoParametersSingle& p,
    int match_metric,
    float second_best_min_distance_factor = 0,
    CUDABuffer_<float>* best_inv_depth_map = nullptr);

// Multiple-image variant
void PatchMatchPropagationStepCUDA(
    const StereoParametersMulti& p,
    int match_metric,
    float second_best_min_distance_factor = 0,
    CUDABuffer_<float>* best_inv_depth_map = nullptr);


// Image-pair variant
void PatchMatchDiscreteRefinementStepCUDA(
    const StereoParametersSingle& p,
    int match_metric,
    int num_steps,
    float range_factor);

// Multiple-image variant
void PatchMatchDiscreteRefinementStepCUDA(
    const StereoParametersMulti& p,
    int match_metric,
    int num_steps,
    float range_factor);


// Image-pair variant
void PatchMatchLeftRightConsistencyCheckCUDA(
    const StereoParametersSingle& p,
    float lr_consistency_factor_threshold,
    const CUDABuffer_<float>& lr_consistency_inv_depth,
    CUDABuffer_<float>* inv_depth_map_out);

// Multiple-image variant
void PatchMatchConsistencyCheckCUDA(
    const StereoParametersMulti& p,
    float lr_consistency_factor_threshold,
    float min_required_amount,
    CUDABuffer_<float>* consistency_inv_depth,
    CUDABuffer_<float>* inv_depth_map_out,
    CUDABuffer_<uchar3>* filter_reasons);


// Image-pair variant
void PatchMatchFilterOutliersCUDA(
    const StereoParametersSingle& p,
    float min_inv_depth,
    float required_range_min_depth,
    float required_range_max_depth,
    const CUDAMatrix3x4& reference_tr_stereo,
    CUDABuffer_<float>* inv_depth_map_out,
    float cost_threshold,
    float epipolar_gradient_threshold,
    float min_cos_angle,
    CUDABuffer_<float>* second_best_costs,
    float second_best_min_cost_factor);

// Multiple-image variant
void PatchMatchFilterOutliersCUDA(
    const StereoParametersMulti& p,
    float min_inv_depth,
    float required_range_min_depth,
    float required_range_max_depth,
    const CUDAMatrix3x4* reference_tr_stereo,
    CUDABuffer_<float>* inv_depth_map_out,
    float cost_threshold,
    float epipolar_gradient_threshold,
    float min_cos_angle,
    CUDABuffer_<float>* second_best_costs,
    float second_best_min_cost_factor,
    CUDABuffer_<uchar3>* filter_reasons);


void MedianFilterDepthMap3x3CUDA(
    cudaStream_t stream,
    int context_radius,
    CUDABuffer_<float>* inv_depth_map,
    CUDABuffer_<float>* inv_depth_map_out,
    CUDABuffer_<float>* costs,
    CUDABuffer_<float>* costs_out,
    CUDABuffer_<float>* second_best_costs,
    CUDABuffer_<float>* second_best_costs_out);

void BilateralFilterCUDA(
    cudaStream_t stream,
    float sigma_xy,
    float sigma_value,
    float radius_factor,
    const CUDABuffer_<float>& inv_depth_map,
    CUDABuffer_<float>* inv_depth_map_out);

void FillHolesCUDA(
    cudaStream_t stream,
    const CUDABuffer_<float>& inv_depth_map,
    CUDABuffer_<float>* inv_depth_map_out);

}
