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

__global__ void PatchMatchLeftRightConsistencyCheckCUDAKernel(
    const StereoParametersSingleCUDA p,
    float lr_consistency_factor_threshold,
    const CUDABuffer_<float> lr_consistency_inv_depth,
    CUDABuffer_<float> inv_depth_map_out) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  constexpr float kInvalidInvDepth = 0;
  
  if (x >= p.context_radius &&
      y >= p.context_radius &&
      x < p.inv_depth_map.width() - p.context_radius &&
      y < p.inv_depth_map.height() - p.context_radius) {
    float inv_depth = p.inv_depth_map(y, x);
    float depth = 1 / inv_depth;
    
    float2 center_nxy = p.reference_unprojection_lookup.UnprojectPoint(x, y);
    float3 reference_point = make_float3(depth * center_nxy.x, depth * center_nxy.y, depth);
    
    float3 pnxy = p.stereo_tr_reference * reference_point;
    if (pnxy.z <= 0.f) {
      inv_depth_map_out(y, x) = kInvalidInvDepth;
      return;
    }
    
    const float2 pxy = p.stereo_camera.Project(pnxy);
    if (pxy.x < p.context_radius ||
        pxy.y < p.context_radius ||
        pxy.x >= p.stereo_camera.width - 1 - p.context_radius ||
        pxy.y >= p.stereo_camera.height - 1 - p.context_radius ||
        (p.mask.address() && p.mask(pxy.y, pxy.x) == 0)) {
      inv_depth_map_out(y, x) = kInvalidInvDepth;
      return;
    }
    
    float lr_check_inv_depth = lr_consistency_inv_depth(pxy.y, pxy.x);
    if (lr_check_inv_depth == kInvalidInvDepth) {
      inv_depth_map_out(y, x) = kInvalidInvDepth;
      return;
    }
    
    float factor = pnxy.z * lr_check_inv_depth;
    if (factor < 1) {
      factor = 1 / factor;
    }
    
    if (factor > lr_consistency_factor_threshold) {
      inv_depth_map_out(y, x) = kInvalidInvDepth;
    } else {
      inv_depth_map_out(y, x) = inv_depth;
    }
  } else if (x < p.inv_depth_map.width() && y < p.inv_depth_map.height()) {
    inv_depth_map_out(y, x) = kInvalidInvDepth;
  }
}

void PatchMatchLeftRightConsistencyCheckCUDA(
    const StereoParametersSingle& p,
    float lr_consistency_factor_threshold,
    const CUDABuffer_<float>& lr_consistency_inv_depth,
    CUDABuffer_<float>* inv_depth_map_out) {
  CHECK_CUDA_NO_ERROR();
  CUDA_AUTO_TUNE_2D(
      PatchMatchLeftRightConsistencyCheckCUDAKernel,
      16, 16,
      p.inv_depth_map.width(), p.inv_depth_map.height(),
      0, p.stream,
      /* kernel parameters */
      StereoParametersSingleCUDA(p),
      lr_consistency_factor_threshold,
      lr_consistency_inv_depth,
      *inv_depth_map_out);
  CHECK_CUDA_NO_ERROR();
}


template <bool kDebugFilterReasons>
__global__ void PatchMatchConsistencyCheckCUDAKernel(
    const StereoParametersMultiCUDA p,
    float lr_consistency_factor_threshold,
    float min_required_amount,
    CUDABuffer_<float>* consistency_inv_depth,
    CUDABuffer_<float> inv_depth_map_out,
    CUDABuffer_<uchar3> filter_reasons) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  constexpr float kInvalidInvDepth = 0;
  
  if (x >= p.context_radius &&
      y >= p.context_radius &&
      x < p.inv_depth_map.width() - p.context_radius &&
      y < p.inv_depth_map.height() - p.context_radius) {
    float inv_depth = p.inv_depth_map(y, x);
    float depth = 1 / inv_depth;
    
    float2 center_nxy = p.reference_unprojection_lookup.UnprojectPoint(x, y);
    float3 reference_point = make_float3(depth * center_nxy.x, depth * center_nxy.y, depth);
    
    int num_consistent = 0;
    int num_in_view = 0;
    for (int s = 0; s < p.num_stereo_images; ++ s) {
      float3 pnxy = p.stereo_tr_reference[s] * reference_point;
      if (pnxy.z <= 0.f) {
        continue;
      }
      
      const float2 pxy = p.stereo_camera.Project(pnxy);
      if (pxy.x < p.context_radius ||
          pxy.y < p.context_radius ||
          pxy.x >= p.stereo_camera.width - 1 - p.context_radius ||
          pxy.y >= p.stereo_camera.height - 1 - p.context_radius ||
          (p.mask.address() && p.mask(pxy.y, pxy.x) == 0)) {
        continue;
      }
      
      float lr_check_inv_depth = consistency_inv_depth[s](pxy.y, pxy.x);
      if (lr_check_inv_depth == kInvalidInvDepth) {
        continue;
      }
      
      ++ num_in_view;
      
      float factor = pnxy.z * lr_check_inv_depth;
      if (factor < 1) {
        factor = 1 / factor;
      }
      
      if (factor <= lr_consistency_factor_threshold) {
        ++ num_consistent;
      }
    }
    
    if (num_consistent < ::ceil(min_required_amount * num_in_view)) {
      inv_depth_map_out(y, x) = kInvalidInvDepth;
      if (kDebugFilterReasons) {
        filter_reasons(y, x) = make_uchar3(0, 0, 255);
      }
    } else {
      inv_depth_map_out(y, x) = inv_depth;
      if (kDebugFilterReasons) {
        filter_reasons(y, x) = make_uchar3(0, 0, 0);
      }
    }
  } else if (x < p.inv_depth_map.width() && y < p.inv_depth_map.height()) {
    inv_depth_map_out(y, x) = kInvalidInvDepth;
    if (kDebugFilterReasons) {
      filter_reasons(y, x) = make_uchar3(127, 127, 127);
    }
  }
}

void PatchMatchConsistencyCheckCUDA(
    const StereoParametersMulti& p,
    float lr_consistency_factor_threshold,
    float min_required_amount,
    CUDABuffer_<float>* consistency_inv_depth,
    CUDABuffer_<float>* inv_depth_map_out,
    CUDABuffer_<uchar3>* filter_reasons) {
  CHECK_CUDA_NO_ERROR();
  bool have_filter_reasons = filter_reasons != nullptr;
  COMPILE_OPTION(have_filter_reasons, CUDA_AUTO_TUNE_2D_TEMPLATED(
      PatchMatchConsistencyCheckCUDAKernel,
      16, 16,
      p.inv_depth_map.width(), p.inv_depth_map.height(),
      0, p.stream,
      TEMPLATE_ARGUMENTS(_have_filter_reasons),
      /* kernel parameters */
      StereoParametersMultiCUDA(p),
      lr_consistency_factor_threshold,
      min_required_amount,
      consistency_inv_depth,
      *inv_depth_map_out,
      filter_reasons ? *filter_reasons : CUDABuffer_<uchar3>()));
  CHECK_CUDA_NO_ERROR();
}

}
