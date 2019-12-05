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

__global__ void PatchMatchFilterOutliersCUDAKernel(
    const StereoParametersSingleCUDA p,
    float min_inv_depth,
    float required_range_min_depth,
    float required_range_max_depth,
    const CUDAMatrix3x4 reference_tr_stereo,
    CUDABuffer_<float> inv_depth_map_out,
    float cost_threshold,
    float epipolar_gradient_threshold,
    float min_cos_angle,
    CUDABuffer_<float> second_best_costs,
    float second_best_min_cost_factor) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  const float kInvalidInvDepth = 0;
  
  if (x >= p.context_radius &&
      y >= p.context_radius &&
      x < p.inv_depth_map.width() - p.context_radius &&
      y < p.inv_depth_map.height() - p.context_radius) {
    if (!(p.costs(y, x) <= cost_threshold) ||  // includes NaNs
        !(p.inv_depth_map(y, x) > min_inv_depth)) {
      inv_depth_map_out(y, x) = kInvalidInvDepth;
    } else {
      // If there is another depth value with similar cost, reject the depth
      // estimate as ambiguous.
      if (second_best_min_cost_factor > 1) {
        if (!(second_best_costs(y, x) >= second_best_min_cost_factor * p.costs(y, x))) {  // includes NaNs
          inv_depth_map_out(y, x) = kInvalidInvDepth;
          return;
        }
      }
      
      // If at the maximum or minimum depth for this pixel the stereo frame
      // would not observe that point, discard the pixel (i.e., enforce that
      // this depth range is observed by both frames).
      // This is to protect against mistakes that often happen when the frames
      // overlap in only a small depth range and the actual depth is not within
      // that range.
      float2 center_nxy = p.reference_unprojection_lookup.UnprojectPoint(x, y);
      float3 range_min_point = make_float3(required_range_min_depth * center_nxy.x, required_range_min_depth * center_nxy.y, required_range_min_depth);
      float3 range_max_point = make_float3(required_range_max_depth * center_nxy.x, required_range_max_depth * center_nxy.y, required_range_max_depth);
      
      float3 rmin_stereo_point = p.stereo_tr_reference * range_min_point;
      if (rmin_stereo_point.z <= 0.f) {
        inv_depth_map_out(y, x) = kInvalidInvDepth;
        return;
      }
      
      const float2 rmin_pxy = p.stereo_camera.Project(rmin_stereo_point);
      if (rmin_pxy.x < p.context_radius ||
          rmin_pxy.y < p.context_radius ||
          rmin_pxy.x >= p.stereo_camera.width - 1 - p.context_radius ||
          rmin_pxy.y >= p.stereo_camera.height - 1 - p.context_radius ||
          (p.mask.address() && p.mask(rmin_pxy.y, rmin_pxy.x) == 0)) {
        inv_depth_map_out(y, x) = kInvalidInvDepth;
        return;
      }
      
      float3 rmax_stereo_point = p.stereo_tr_reference * range_max_point;
      if (rmax_stereo_point.z <= 0.f) {
        inv_depth_map_out(y, x) = kInvalidInvDepth;
        return;
      }
      
      const float2 rmax_pxy = p.stereo_camera.Project(rmax_stereo_point);
      if (rmax_pxy.x < p.context_radius ||
          rmax_pxy.y < p.context_radius ||
          rmax_pxy.x >= p.stereo_camera.width - 1 - p.context_radius ||
          rmax_pxy.y >= p.stereo_camera.height - 1 - p.context_radius ||
          (p.mask.address() && p.mask(rmax_pxy.y, rmax_pxy.x) == 0)) {
        inv_depth_map_out(y, x) = kInvalidInvDepth;
        return;
      }
      
      // Texture filtering: remove pixels with too small gradients along the epipolar line direction in the patch used for matching.
      // TODO: The code below is only valid for the current ZNCC implementation, not SSD or Census!
      float inv_depth = p.inv_depth_map(y, x);
      
      const char2 normal_char = p.normals(y, x);
      float2 normal_xy = make_float2(
          normal_char.x * (1 / 127.f), normal_char.y * (1 / 127.f));
      
      const float normal_z =
          -sqrtf(1.f - normal_xy.x * normal_xy.x - normal_xy.y * normal_xy.y);
      const float depth = 1.f / inv_depth;
      const float plane_d =
          (center_nxy.x * depth) * normal_xy.x +
          (center_nxy.y * depth) * normal_xy.y + depth * normal_z;
      
      float total_gradient_magnitude = 0;
          
      for (int sample = 0; sample < kNumSamples; ++ sample) {
        float dx = p.context_radius * kSamplesCUDA[sample][0];
        float dy = p.context_radius * kSamplesCUDA[sample][1];
        
        int ix = ::max(0, ::min(static_cast<int>(p.inv_depth_map.width()) - 1, static_cast<int>(x + dx)));
        int iy = ::max(0, ::min(static_cast<int>(p.inv_depth_map.height()) - 1, static_cast<int>(y + dy)));
        if (p.mask.address() && p.mask(iy, ix) == 0) {
          total_gradient_magnitude = -1;
          break;
        }
        
        float2 nxy = p.reference_unprojection_lookup.UnprojectPoint(x + dx, y + dy);  // NOTE: This is only approximate (bilinear interpolation of exact values sampled at pixel centers).
        float plane_depth = CalculatePlaneDepth2(plane_d, normal_xy, normal_z, nxy.x, nxy.y);
        
        float3 original_reference_point = make_float3(nxy.x * plane_depth, nxy.y * plane_depth, plane_depth);
        float3 original_stereo_point = p.stereo_tr_reference * original_reference_point;
        constexpr float kShiftZ = 0.01f;
        float3 shifted_stereo_point = make_float3(original_stereo_point.x, original_stereo_point.y, original_stereo_point.z + kShiftZ);
        float3 shifted_reference_point = reference_tr_stereo * shifted_stereo_point;
        
        const float2 shifted_projection = p.stereo_camera.Project(shifted_reference_point);
        float2 epipolar_direction = make_float2(shifted_projection.x - 0.5f - (x + dx),
                                                shifted_projection.y - 0.5f - (y + dy));
        
        float length = sqrtf(epipolar_direction.x * epipolar_direction.x + epipolar_direction.y * epipolar_direction.y);
        epipolar_direction = make_float2(epipolar_direction.x / length, epipolar_direction.y / length);  // Normalize to length of 1 pixel
        
        float reference_value = 255.f * tex2D<float>(p.reference_texture, x + dx + 0.5f, y + dy + 0.5f);
        float shifted_reference_value = 255.f * tex2D<float>(p.reference_texture, x + dx + 0.5f + epipolar_direction.x, y + dy + 0.5f + epipolar_direction.y);
        
        total_gradient_magnitude += fabs(shifted_reference_value - reference_value);
      }
      
      if (total_gradient_magnitude < epipolar_gradient_threshold) {
        inv_depth_map_out(y, x) = kInvalidInvDepth;
        return;
      }
      
      // Angle filtering.
      // Estimate the surface normal from the depth map.
      float center_depth = 1.f / p.inv_depth_map(y, x);
      float right_depth = 1.f / p.inv_depth_map(y, x + 1);
      float left_depth = 1.f / p.inv_depth_map(y, x - 1);
      float bottom_depth = 1.f / p.inv_depth_map(y + 1, x);
      float top_depth = 1.f / p.inv_depth_map(y - 1, x);
      
      float2 left_nxy = p.reference_unprojection_lookup.UnprojectPoint(x - 1, y);
      float3 left_point = make_float3(left_depth * left_nxy.x, left_depth * left_nxy.y, left_depth);
      
      float2 right_nxy = p.reference_unprojection_lookup.UnprojectPoint(x + 1, y);
      float3 right_point = make_float3(right_depth * right_nxy.x, right_depth * right_nxy.y, right_depth);
      
      float2 top_nxy = p.reference_unprojection_lookup.UnprojectPoint(x, y - 1);
      float3 top_point = make_float3(top_depth * top_nxy.x, top_depth * top_nxy.y, top_depth);
      
      float2 bottom_nxy = p.reference_unprojection_lookup.UnprojectPoint(x, y + 1);
      float3 bottom_point = make_float3(bottom_depth * bottom_nxy.x, bottom_depth * bottom_nxy.y, bottom_depth);
      
      float3 center_point = make_float3(center_depth * center_nxy.x, center_depth * center_nxy.y, center_depth);
      
      constexpr float kRatioThreshold = 2.f;
      constexpr float kRatioThresholdSquared = kRatioThreshold * kRatioThreshold;
      
      float left_dist_squared = SquaredLength(left_point - center_point);
      float right_dist_squared = SquaredLength(right_point - center_point);
      float left_right_ratio = left_dist_squared / right_dist_squared;
      float3 left_to_right;
      if (left_right_ratio < kRatioThresholdSquared &&
          left_right_ratio > 1.f / kRatioThresholdSquared) {
        left_to_right = right_point - left_point;
      } else if (left_dist_squared < right_dist_squared) {
        left_to_right = center_point - left_point;
      } else {  // left_dist_squared >= right_dist_squared
        left_to_right = right_point - center_point;
      }
      
      float bottom_dist_squared = SquaredLength(bottom_point - center_point);
      float top_dist_squared = SquaredLength(top_point - center_point);
      float bottom_top_ratio = bottom_dist_squared / top_dist_squared;
      float3 bottom_to_top;
      if (bottom_top_ratio < kRatioThresholdSquared &&
          bottom_top_ratio > 1.f / kRatioThresholdSquared) {
        bottom_to_top = top_point - bottom_point;
      } else if (bottom_dist_squared < top_dist_squared) {
        bottom_to_top = center_point - bottom_point;
      } else {  // bottom_dist_squared >= top_dist_squared
        bottom_to_top = top_point - center_point;
      }
      
      float3 normal;
      CrossProduct(left_to_right, bottom_to_top, &normal);
      
      // Apply angle threshold.
      const float normal_length = Norm(normal);
      const float point_distance = Norm(center_point);
      const float view_cos_angle = Dot(normal, center_point) / (normal_length * point_distance);
      
      if (view_cos_angle > min_cos_angle) {
        inv_depth_map_out(y, x) = kInvalidInvDepth;
      } else {
        inv_depth_map_out(y, x) = p.inv_depth_map(y, x);
      }
    }
  } else if (x < p.inv_depth_map.width() && y < p.inv_depth_map.height()) {
    inv_depth_map_out(y, x) = kInvalidInvDepth;
  }
}

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
    float second_best_min_cost_factor) {
  CHECK_CUDA_NO_ERROR();
  CUDA_AUTO_TUNE_2D(
      PatchMatchFilterOutliersCUDAKernel,
      16, 16,
      p.inv_depth_map.width(), p.inv_depth_map.height(),
      0, p.stream,
      /* kernel parameters */
      StereoParametersSingleCUDA(p),
      min_inv_depth,
      required_range_min_depth,
      required_range_max_depth,
      reference_tr_stereo,
      *inv_depth_map_out,
      cost_threshold,
      epipolar_gradient_threshold,
      min_cos_angle,
      *second_best_costs,
      second_best_min_cost_factor);
  CHECK_CUDA_NO_ERROR();
}


template <bool kDebugFilterReasons>
__global__ void PatchMatchFilterOutliersCUDAKernel(
    const StereoParametersMultiCUDA p,
    float min_inv_depth,
    float required_range_min_depth,
    float required_range_max_depth,
    const CUDAMatrix3x4* reference_tr_stereo,
    CUDABuffer_<float> inv_depth_map_out,
    float cost_threshold,
    float epipolar_gradient_threshold,
    float min_cos_angle,
    CUDABuffer_<float> second_best_costs,
    float second_best_min_cost_factor,
    CUDABuffer_<uchar3> filter_reasons) {
  // List of filter reasons with debug color:
  // dark red    (127, 0, 0): The depth exceeds the maximum depth
  // dark green  (0, 127, 0): The required depth range is not visible in any stereo image
  // red         (255, 0, 0): The gradients in epipolar line directions are too small for all stereo images
  //                          (note: this only uses image-bounds visibility checking in the stereo images,
  //                           so it may incorrectly take images into account where the point is occluded)
  // dark yellow (140, 140, 0): Angle check failed
  // gray        (127, 127, 127): Pixel is too close to the image borders (closer than context radius)
  // blue        (0, 0, 255): Consistency check failed.
  // green       (0, 255, 0): Connected component too small.
  // black       (0, 0, 0):   The pixel passed the filters.
  
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  const float kInvalidInvDepth = 0;
  
  if (x >= p.context_radius &&
      y >= p.context_radius &&
      x < p.inv_depth_map.width() - p.context_radius &&
      y < p.inv_depth_map.height() - p.context_radius) {
    if (!(p.inv_depth_map(y, x) > min_inv_depth)) {  // includes NaNs
      if (kDebugFilterReasons) {
        if (p.inv_depth_map(y, x) != kInvalidInvDepth) {
          filter_reasons(y, x) = make_uchar3(127, 0, 0);
        }
      }
      inv_depth_map_out(y, x) = kInvalidInvDepth;
    } else {
      // If there is another depth value with similar cost, reject the depth
      // estimate as ambiguous.
//       if (second_best_min_cost_factor > 1) {
//         if (!(second_best_costs(y, x) >= second_best_min_cost_factor * costs(y, x))) {  // includes NaNs
//           inv_depth_map_out(y, x) = kInvalidInvDepth;
//           return;
//         }
//       }
      
      // If at the maximum or minimum depth for this pixel the stereo frame
      // would not observe that point, discard the pixel (i.e., enforce that
      // this depth range is observed by both frames).
      // This is to protect against mistakes that often happen when the frames
      // overlap in only a small depth range and the actual depth is not within
      // that range.
      float2 center_nxy = p.reference_unprojection_lookup.UnprojectPoint(x, y);
      float3 range_min_point = make_float3(required_range_min_depth * center_nxy.x, required_range_min_depth * center_nxy.y, required_range_min_depth);
      float3 range_max_point = make_float3(required_range_max_depth * center_nxy.x, required_range_max_depth * center_nxy.y, required_range_max_depth);
      
      bool valid = false;
      for (int s = 0; s < p.num_stereo_images; ++ s) {
        float3 rmin_stereo_point = p.stereo_tr_reference[s] * range_min_point;
        if (rmin_stereo_point.z <= 0.f) {
          continue;
        }
        
        const float2 rmin_pxy = p.stereo_camera.Project(rmin_stereo_point);
        if (rmin_pxy.x < p.context_radius ||
            rmin_pxy.y < p.context_radius ||
            rmin_pxy.x >= p.stereo_camera.width - 1 - p.context_radius ||
            rmin_pxy.y >= p.stereo_camera.height - 1 - p.context_radius ||
            (p.mask.address() && p.mask(rmin_pxy.y, rmin_pxy.x) == 0)) {
          continue;
        }
        
        float3 rmax_stereo_point = p.stereo_tr_reference[s] * range_max_point;
        if (rmax_stereo_point.z <= 0.f) {
          continue;
        }
        
        const float2 rmax_pxy = p.stereo_camera.Project(rmax_stereo_point);
        if (rmax_pxy.x < p.context_radius ||
            rmax_pxy.y < p.context_radius ||
            rmax_pxy.x >= p.stereo_camera.width - 1 - p.context_radius ||
            rmax_pxy.y >= p.stereo_camera.height - 1 - p.context_radius ||
            (p.mask.address() && p.mask(rmax_pxy.y, rmax_pxy.x) == 0)) {
          continue;
        }
        
        valid = true;
        break;
      }
      if (!valid) {
        inv_depth_map_out(y, x) = kInvalidInvDepth;
        if (kDebugFilterReasons) {
          filter_reasons(y, x) = make_uchar3(0, 127, 0);
        }
        return;
      }
      
      // TODO: Texture filtering is currently not implemented for the multi-image case
      // Texture filtering: remove pixels with too small gradients along the epipolar line direction in the patch used for matching.
      // TODO: The code below is only valid for the current ZNCC implementation, not SSD or Census!
      float inv_depth = p.inv_depth_map(y, x);
      
      const char2 normal_char = p.normals(y, x);
      float2 normal_xy = make_float2(
          normal_char.x * (1 / 127.f), normal_char.y * (1 / 127.f));
      
      const float normal_z =
          -sqrtf(1.f - normal_xy.x * normal_xy.x - normal_xy.y * normal_xy.y);
      const float depth = 1.f / inv_depth;
      const float plane_d =
          (center_nxy.x * depth) * normal_xy.x +
          (center_nxy.y * depth) * normal_xy.y + depth * normal_z;
      
      valid = false;
      for (int s = 0; s < p.num_stereo_images; ++ s) {
        float total_gradient_magnitude = 0;
            
        for (int sample = 0; sample < kNumSamples; ++ sample) {
          float dx = p.context_radius * kSamplesCUDA[sample][0];
          float dy = p.context_radius * kSamplesCUDA[sample][1];
          
          if (s == 0) {
            int ix = ::max(0, ::min(static_cast<int>(p.inv_depth_map.width() - 1), static_cast<int>(x + dx)));
            int iy = ::max(0, ::min(static_cast<int>(p.inv_depth_map.height() - 1), static_cast<int>(y + dy)));
            if (p.mask.address() && p.mask(iy, ix) == 0) {
              inv_depth_map_out(y, x) = kInvalidInvDepth;
              if (kDebugFilterReasons) {
                filter_reasons(y, x) = make_uchar3(127, 127, 127);
              }
              return;
            }
          }
          
          float2 nxy = p.reference_unprojection_lookup.UnprojectPoint(x + dx, y + dy);  // NOTE: This is only approximate (bilinear interpolation of exact values sampled at pixel centers).
          float plane_depth = CalculatePlaneDepth2(plane_d, normal_xy, normal_z, nxy.x, nxy.y);
          
          float3 original_reference_point = make_float3(nxy.x * plane_depth, nxy.y * plane_depth, plane_depth);
          float3 original_stereo_point = p.stereo_tr_reference[s] * original_reference_point;
          if (original_stereo_point.z <= 0) {
            continue;
          }
          const float2 stereo_projection = p.stereo_camera.Project(original_stereo_point);
          if (stereo_projection.x < p.context_radius ||
              stereo_projection.y < p.context_radius ||
              stereo_projection.x >= p.stereo_camera.width - 1 - p.context_radius ||
              stereo_projection.y >= p.stereo_camera.height - 1 - p.context_radius ||
              (p.mask.address() && p.mask(stereo_projection.y, stereo_projection.x) == 0)) {
            continue;
          }
          
          constexpr float kShiftZ = 0.01f;
          float3 shifted_stereo_point = make_float3(original_stereo_point.x, original_stereo_point.y, original_stereo_point.z + kShiftZ);
          float3 shifted_reference_point = reference_tr_stereo[s] * shifted_stereo_point;
          
          const float2 shifted_projection = p.stereo_camera.Project(shifted_reference_point);
          float2 epipolar_direction = make_float2(shifted_projection.x - 0.5f - (x + dx),
                                                  shifted_projection.y - 0.5f - (y + dy));
          
          float length = sqrtf(epipolar_direction.x * epipolar_direction.x + epipolar_direction.y * epipolar_direction.y);
          epipolar_direction = make_float2(epipolar_direction.x / length, epipolar_direction.y / length);  // Normalize to length of 1 pixel
          
          float reference_value = 255.f * tex2D<float>(p.reference_texture, x + dx + 0.5f, y + dy + 0.5f);
          float shifted_reference_value = 255.f * tex2D<float>(p.reference_texture, x + dx + 0.5f + epipolar_direction.x, y + dy + 0.5f + epipolar_direction.y);
          
          total_gradient_magnitude += fabs(shifted_reference_value - reference_value);
        }
        
        if (total_gradient_magnitude >= epipolar_gradient_threshold) {
          valid = true;
          break;
        }
      }
      if (!valid) {
        inv_depth_map_out(y, x) = kInvalidInvDepth;
        if (kDebugFilterReasons) {
          filter_reasons(y, x) = make_uchar3(255, 0, 0);
        }
        return;
      }
      
      // Angle filtering.
      // Estimate the surface normal from the depth map.
      float center_depth = 1.f / p.inv_depth_map(y, x);
      float right_depth = 1.f / p.inv_depth_map(y, x + 1);
      float left_depth = 1.f / p.inv_depth_map(y, x - 1);
      float bottom_depth = 1.f / p.inv_depth_map(y + 1, x);
      float top_depth = 1.f / p.inv_depth_map(y - 1, x);
      
      float2 left_nxy = p.reference_unprojection_lookup.UnprojectPoint(x - 1, y);
      float3 left_point = make_float3(left_depth * left_nxy.x, left_depth * left_nxy.y, left_depth);
      
      float2 right_nxy = p.reference_unprojection_lookup.UnprojectPoint(x + 1, y);
      float3 right_point = make_float3(right_depth * right_nxy.x, right_depth * right_nxy.y, right_depth);
      
      float2 top_nxy = p.reference_unprojection_lookup.UnprojectPoint(x, y - 1);
      float3 top_point = make_float3(top_depth * top_nxy.x, top_depth * top_nxy.y, top_depth);
      
      float2 bottom_nxy = p.reference_unprojection_lookup.UnprojectPoint(x, y + 1);
      float3 bottom_point = make_float3(bottom_depth * bottom_nxy.x, bottom_depth * bottom_nxy.y, bottom_depth);
      
      float3 center_point = make_float3(center_depth * center_nxy.x, center_depth * center_nxy.y, center_depth);
      
      constexpr float kRatioThreshold = 2.f;
      constexpr float kRatioThresholdSquared = kRatioThreshold * kRatioThreshold;
      
      float left_dist_squared = SquaredLength(left_point - center_point);
      float right_dist_squared = SquaredLength(right_point - center_point);
      float left_right_ratio = left_dist_squared / right_dist_squared;
      float3 left_to_right;
      if (left_right_ratio < kRatioThresholdSquared &&
          left_right_ratio > 1.f / kRatioThresholdSquared) {
        left_to_right = right_point - left_point;
      } else if (left_dist_squared < right_dist_squared) {
        left_to_right = center_point - left_point;
      } else {  // left_dist_squared >= right_dist_squared
        left_to_right = right_point - center_point;
      }
      
      float bottom_dist_squared = SquaredLength(bottom_point - center_point);
      float top_dist_squared = SquaredLength(top_point - center_point);
      float bottom_top_ratio = bottom_dist_squared / top_dist_squared;
      float3 bottom_to_top;
      if (bottom_top_ratio < kRatioThresholdSquared &&
          bottom_top_ratio > 1.f / kRatioThresholdSquared) {
        bottom_to_top = top_point - bottom_point;
      } else if (bottom_dist_squared < top_dist_squared) {
        bottom_to_top = center_point - bottom_point;
      } else {  // bottom_dist_squared >= top_dist_squared
        bottom_to_top = top_point - center_point;
      }
      
      float3 normal;
      CrossProduct(left_to_right, bottom_to_top, &normal);
      
      // Apply angle threshold.
      const float normal_length = Norm(normal);
      const float point_distance = Norm(center_point);
      const float view_cos_angle = Dot(normal, center_point) / (normal_length * point_distance);
      
      if (view_cos_angle > min_cos_angle) {
        inv_depth_map_out(y, x) = kInvalidInvDepth;
        if (kDebugFilterReasons) {
          filter_reasons(y, x) = make_uchar3(140, 140, 0);
        }
      } else {
        inv_depth_map_out(y, x) = p.inv_depth_map(y, x);
        if (kDebugFilterReasons) {
          filter_reasons(y, x) = make_uchar3(0, 0, 0);
        }
      }
    }
  } else if (x < p.inv_depth_map.width() && y < p.inv_depth_map.height()) {
    inv_depth_map_out(y, x) = kInvalidInvDepth;
    if (kDebugFilterReasons) {
      filter_reasons(y, x) = make_uchar3(127, 127, 127);
    }
  }
}

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
    CUDABuffer_<uchar3>* filter_reasons) {
  CHECK_CUDA_NO_ERROR();
  bool have_filter_reasons = filter_reasons != nullptr;
  COMPILE_OPTION(have_filter_reasons, CUDA_AUTO_TUNE_2D_TEMPLATED(
      PatchMatchFilterOutliersCUDAKernel,
      16, 16,
      p.inv_depth_map.width(), p.inv_depth_map.height(),
      0, p.stream,
      TEMPLATE_ARGUMENTS(_have_filter_reasons),
      /* kernel parameters */
      StereoParametersMultiCUDA(p),
      min_inv_depth,
      required_range_min_depth,
      required_range_max_depth,
      reference_tr_stereo,
      *inv_depth_map_out,
      cost_threshold,
      epipolar_gradient_threshold,
      min_cos_angle,
      *second_best_costs,
      second_best_min_cost_factor,
      filter_reasons ? *filter_reasons : CUDABuffer_<uchar3>()));
  CHECK_CUDA_NO_ERROR();
}

}
