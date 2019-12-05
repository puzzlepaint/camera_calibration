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

#include "libvis/cuda/cuda_buffer.cuh"
#include "libvis/cuda/cuda_matrix.cuh"
#include "libvis/cuda/cuda_unprojection_lookup.cuh"
#include "libvis/cuda/pixel_corner_projector.cuh"
#include "libvis/cuda/patch_match_stereo_samples.cuh"
#include "libvis/cuda/patch_match_stereo_util.cuh"
#include "libvis/libvis.h"

namespace vis {

constexpr float kMinInvDepth = 1e-5f;  // TODO: Make parameter


__forceinline__ __device__ float SampleAtProjectedPosition(
    const float x, const float y, const float z,
    const PixelCornerProjector_& projector,
    const CUDABuffer_<u8>& mask,
    const CUDAMatrix3x4& stereo_tr_reference,
    cudaTextureObject_t stereo_texture) {
  float3 pnxy = stereo_tr_reference * make_float3(x, y, z);
  if (pnxy.z <= 0.f) {
    return CUDART_NAN_F;
  }
  
  const float2 pxy = projector.Project(pnxy);
  
  if (pxy.x < 0.5f ||
      pxy.y < 0.5f ||
      pxy.x >= projector.width - 0.5f ||
      pxy.y >= projector.height - 0.5f ||
      (mask.address() && mask(pxy.y, pxy.x) == 0)) {
    return CUDART_NAN_F;
  } else {
    return 255.0f * tex2D<float>(stereo_texture, pxy.x, pxy.y);
  }
}


__forceinline__ __device__ float CalculatePlaneDepth2(
    float d, const float2& normal_xy, float normal_z,
    float query_x, float query_y) {
  return d / (query_x * normal_xy.x + query_y * normal_xy.y + normal_z);
}

__forceinline__ __device__ float CalculatePlaneInvDepth2(
    float d, const float2& normal_xy, float normal_z,
    float query_x, float query_y) {
  return (query_x * normal_xy.x + query_y * normal_xy.y + normal_z) / d;
}


// __forceinline__ __device__ float ComputeCostsSSD(
//     int x, int y,
//     const float2& normal_xy,
//     const float inv_depth,
//     const float context_radius,
//     const CUDAUnprojectionLookup2D_& unprojector,
//     const CUDABuffer_<u8>& reference_image,
//     const CUDAMatrix3x4& stereo_tr_reference,
//     const PixelCornerProjector_& projector,
//     const CUDABuffer_<u8>& mask,
//     cudaTextureObject_t stereo_image) {
//   if (inv_depth < kMinInvDepth) {
//     return CUDART_NAN_F;
//   }
//   
//   const float normal_z =
//       -sqrtf(1.f - normal_xy.x * normal_xy.x - normal_xy.y * normal_xy.y);
//   const float depth = 1.f / inv_depth;
//   const float2 center_nxy =
//       unprojector.UnprojectPoint(x, y);
//   const float plane_d =
//       (center_nxy.x * depth) * normal_xy.x +
//       (center_nxy.y * depth) * normal_xy.y + depth * normal_z;
//   
//   float cost = 0;
//   
//   #pragma unroll
//   for (int dy = -context_radius; dy <= context_radius; ++ dy) {
//     #pragma unroll
//     for (int dx = -context_radius; dx <= context_radius; ++ dx) {
//       float2 nxy = unprojector.UnprojectPoint(x + dx, y + dy);
//       float plane_depth = CalculatePlaneDepth2(plane_d, normal_xy, normal_z, nxy.x, nxy.y);
//       nxy.x *= plane_depth;
//       nxy.y *= plane_depth;
//       
//       float sample =
//             SampleAtProjectedPosition(nxy.x, nxy.y, plane_depth,
//                                       projector,
//                                       mask,
//                                       stereo_tr_reference,
//                                       stereo_image);
//       
//       const float diff = sample - reference_image(y + dy, x + dx);
//       cost += diff * diff;
//     }
//   }
//   
//   return cost;
// }


// Computes 0.5f * (1 - ZNCC), so that the result can be used
// as a cost value with range [0; 1].
__forceinline__ __device__ float ComputeZNCCBasedCost(
    const int num_samples,
    const float sum_a,
    const float squared_sum_a,
    const float sum_b,
    const float squared_sum_b,
    const float product_sum) {
  const float normalizer = 1.0f / num_samples;

  const float numerator =
      product_sum - normalizer * (sum_a * sum_b);
  const float denominator_reference =
      squared_sum_a - normalizer * sum_a * sum_a;
  const float denominator_other =
      squared_sum_b - normalizer * sum_b * sum_b;
  
  // NOTE: Using a threshold on homogeneous patches is required here since
  //       otherwise the optimum might be a noisy value in a homogeneous area.
  constexpr float kHomogeneousThreshold = 0.1f;
  if (denominator_reference < kHomogeneousThreshold ||
      denominator_other < kHomogeneousThreshold) {
    return 1.0f;
  } else {
    return 0.5f * (1.0f - numerator *
        rsqrtf(denominator_reference * denominator_other));
  }
}


__forceinline__ __device__ float ComputeCostsZNCC(
    int x, int y,
    const float2& normal_xy,
    const float inv_depth,
    const float context_radius,
    const CUDAUnprojectionLookup2D_& unprojector,
    const CUDABuffer_<u8>& reference_image,
    cudaTextureObject_t reference_texture,
    const CUDAMatrix3x4& stereo_tr_reference,
    const PixelCornerProjector_& projector,
    const CUDABuffer_<u8>& mask,
    cudaTextureObject_t stereo_image) {
  if (inv_depth < kMinInvDepth) {
    return CUDART_NAN_F;
  }
  
  const float normal_z =
      -sqrtf(1.f - normal_xy.x * normal_xy.x - normal_xy.y * normal_xy.y);
  const float depth = 1.f / inv_depth;
  const float2 center_nxy =
      unprojector.UnprojectPoint(x, y);
  const float plane_d =
      (center_nxy.x * depth) * normal_xy.x +
      (center_nxy.y * depth) * normal_xy.y + depth * normal_z;
  
  float sum_a = 0;
  float squared_sum_a = 0;
  float sum_b = 0;
  float squared_sum_b = 0;
  float product_sum = 0;
  
  for (int sample = 0; sample < kNumSamples; ++ sample) {
    float dx = context_radius * kSamplesCUDA[sample][0];
    float dy = context_radius * kSamplesCUDA[sample][1];
    
    float2 nxy = unprojector.UnprojectPoint(x + dx, y + dy);  // NOTE: This is only approximate (bilinear interpolation of exact values sampled at pixel centers).
    float plane_depth = CalculatePlaneDepth2(plane_d, normal_xy, normal_z, nxy.x, nxy.y);
    nxy.x *= plane_depth;
    nxy.y *= plane_depth;
    
    float stereo_value =
          SampleAtProjectedPosition(nxy.x, nxy.y, plane_depth,
                                    projector,
                                    mask,
                                    stereo_tr_reference,
                                    stereo_image);
    
    sum_a += stereo_value;
    squared_sum_a += stereo_value * stereo_value;
    
    float reference_value = 255.f * tex2D<float>(reference_texture, x + dx + 0.5f, y + dy + 0.5f);
    
    sum_b += reference_value;
    squared_sum_b += reference_value * reference_value;
    
    product_sum += stereo_value * reference_value;
  }
  
  return ComputeZNCCBasedCost(
      kNumSamples, sum_a, squared_sum_a, sum_b, squared_sum_b, product_sum);
}


// __forceinline__ __device__ float ComputeCostsCensus(
//     int x, int y,
//     const float2& normal_xy,
//     const float inv_depth,
//     const float context_radius,
//     const CUDAUnprojectionLookup2D_& unprojector,
//     const CUDABuffer_<u8>& reference_image,
//     const CUDAMatrix3x4& stereo_tr_reference,
//     const PixelCornerProjector_& projector,
//     cudaTextureObject_t stereo_image) {
//   if (inv_depth < kMinInvDepth) {
//     return CUDART_NAN_F;
//   }
//   
//   const float normal_z =
//       -sqrtf(1.f - normal_xy.x * normal_xy.x - normal_xy.y * normal_xy.y);
//   const float depth = 1.f / inv_depth;
//   const float2 center_nxy =
//       unprojector.UnprojectPoint(x, y);
//   const float plane_d =
//       (center_nxy.x * depth) * normal_xy.x +
//       (center_nxy.y * depth) * normal_xy.y + depth * normal_z;
//   
//   float stereo_center_value =
//       SampleAtProjectedPosition(center_nxy.x * depth, center_nxy.y * depth, depth,
//                                 projector,
//                                 mask,
//                                 stereo_tr_reference,
//                                 stereo_image);
//   u8 reference_center_value = reference_image(y, x);
//   
//   float cost = 0;
//   
//   constexpr int kSpreadFactor = 2;  // TODO: Make parameter
//   
//   #pragma unroll
//   for (int dy = -kSpreadFactor * context_radius; dy <= kSpreadFactor * context_radius; dy += kSpreadFactor) {
//     #pragma unroll
//     for (int dx = -kSpreadFactor * context_radius; dx <= kSpreadFactor * context_radius; dx += kSpreadFactor) {
//       if (dx == 0 && dy == 0) {
//         continue;
//       }
//       if (x + dx < 0 ||
//           y + dy < 0 ||
//           x + dx >= reference_image.width() ||
//           y + dy >= reference_image.height()) {
//         continue;
//       }
//       
//       float2 nxy = unprojector.UnprojectPoint(x + dx, y + dy);
//       float plane_depth = CalculatePlaneDepth2(plane_d, normal_xy, normal_z, nxy.x, nxy.y);
//       nxy.x *= plane_depth;
//       nxy.y *= plane_depth;
//       
//       float stereo_value =
//             SampleAtProjectedPosition(nxy.x, nxy.y, plane_depth,
//                                       projector,
//                                       mask,
//                                       stereo_tr_reference,
//                                       stereo_image);
//       if (::isnan(stereo_value)) {
//         return CUDART_NAN_F;
//       }
//       int stereo_bit = stereo_value > stereo_center_value;
//       
//       u8 reference_value = reference_image(y + dy, x + dx);
//       int reference_bit = reference_value > reference_center_value;
//       
//       cost += stereo_bit != reference_bit;
//     }
//   }
//   
//   return cost;
// }

__forceinline__ __device__ float ComputeCosts(
    int x, int y,
    const float2& normal_xy,
    const float inv_depth,
    int context_radius,
    const CUDAUnprojectionLookup2D_& unprojector,
    const CUDABuffer_<u8>& reference_image,
    cudaTextureObject_t reference_texture,
    const CUDAMatrix3x4& stereo_tr_reference,
    const PixelCornerProjector_& stereo_camera,
    const CUDABuffer_<u8>& mask,
    cudaTextureObject_t stereo_image,
    int match_metric,
    float second_best_min_distance_factor,
    const CUDABuffer_<float>& best_inv_depth_map) {
  if (second_best_min_distance_factor > 0) {
    // Reject estimates which are too close to the best inv depth.
    float best_inv_depth = best_inv_depth_map(y, x);
    float factor = best_inv_depth / inv_depth;
    if (factor < 1) {
      factor = 1 / factor;
    }
    if (factor < second_best_min_distance_factor) {
      return CUDART_NAN_F;
    }
  }
  
  // TODO: Commented out for higher compile speed (and since only ZNCC is consistent with outlier filtering etc.)
//   if (match_metric == kPatchMatchStereo_MatchMetric_SSD) {
//     return ComputeCostsSSD(
//         x, y, normal_xy, inv_depth, context_radius, unprojector, reference_image,
//         stereo_tr_reference, stereo_camera, stereo_image);
//   } else if (match_metric == kPatchMatchStereo_MatchMetric_ZNCC) {
    return ComputeCostsZNCC(
        x, y, normal_xy, inv_depth, context_radius, unprojector, reference_image, reference_texture,
        stereo_tr_reference, stereo_camera, mask, stereo_image);
//   } else {  // if (match_metric == kPatchMatchStereo_MatchMetric_Census) {
//     return ComputeCostsCensus(
//         x, y, normal_xy, inv_depth, context_radius, unprojector, reference_image,
//         stereo_tr_reference, stereo_camera, stereo_image);
//   }
}

__forceinline__ __device__ float ComputeCosts(
    int x, int y,
    const float2& normal_xy,
    const float inv_depth,
    const StereoParametersSingleCUDA& p,
    int match_metric,
    float second_best_min_distance_factor,
    const CUDABuffer_<float>& best_inv_depth_map) {
  return ComputeCosts(
    x, y,
    normal_xy,
    inv_depth,
    p.context_radius,
    p.reference_unprojection_lookup,
    p.reference_image,
    p.reference_texture,
    p.stereo_tr_reference,
    p.stereo_camera,
    p.mask,
    p.stereo_image,
    match_metric,
    second_best_min_distance_factor,
    best_inv_depth_map);
}

/// ComputeCosts() variant for multiple stereo images
__forceinline__ __device__ bool IsCostOfProposedChangeLower(
    int x, int y,
    const float2& normal_xy,
    const float inv_depth,
    const float2& proposed_normal_xy,
    const float proposed_inv_depth,
    const StereoParametersMultiCUDA& p,
    int match_metric,
    float second_best_min_distance_factor,
    const CUDABuffer_<float>& best_inv_depth_map) {
  float old_cost_sum = 0;
  float new_cost_sum = 0;
  
  // TODO: Cache the unprojected points for both states to use them for all stereo images?
  for (int s = 0; s < p.num_stereo_images; ++ s) {
    float old_cost = ComputeCosts(
        x, y,
        normal_xy,
        inv_depth,
        p.context_radius,
        p.reference_unprojection_lookup,
        p.reference_image,
        p.reference_texture,
        p.stereo_tr_reference[s],
        p.stereo_camera,
        p.mask,
        p.stereo_images[s],
        match_metric,
        second_best_min_distance_factor,
        best_inv_depth_map);
    if (::isnan(old_cost)) {
      continue;
    }
    
    float new_cost = ComputeCosts(
        x, y,
        proposed_normal_xy,
        proposed_inv_depth,
        p.context_radius,
        p.reference_unprojection_lookup,
        p.reference_image,
        p.reference_texture,
        p.stereo_tr_reference[s],
        p.stereo_camera,
        p.mask,
        p.stereo_images[s],
        match_metric,
        second_best_min_distance_factor,
        best_inv_depth_map);
    if (::isnan(new_cost)) {
      continue;
    }
    
    old_cost_sum += old_cost;
    new_cost_sum += new_cost;
  }
  
  if (old_cost_sum == 0 && new_cost_sum == 0) {
    // No cost was valid for both states --> no info to base decision on
    return false;
  } else {
    return new_cost_sum < old_cost_sum;
  }
}

}
