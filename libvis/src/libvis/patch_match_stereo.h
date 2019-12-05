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

#include "libvis/camera.h"
#include "libvis/eigen.h"
#include "libvis/image.h"
#include "libvis/libvis.h"
#include "libvis/sophus.h"

namespace vis {

// PatchMatch Stereo implementation for the CPU.
class PatchMatchStereoCPU {
 public:
  enum class MatchMetric {
    // Will not work well on images with differing lighting conditions or
    // specular lighting.
    kSSD = 0,
    
    // Better suited for most applications, however returns very noisy values
    // in homogeneous areas.
    kZNCC = 1
  };
  
  PatchMatchStereoCPU(int width, int height);
  
  void ComputeDepthMap(
      const Camera& reference_camera,
      const Image<u8>& reference_image,
      const SE3f& reference_image_tr_global,
      const Camera& stereo_camera,
      const Image<u8>& stereo_image,
      const SE3f& stereo_image_tr_global,
      Image<float>* inv_depth_map);
  
  // PatchMatch stereo settings accessors
  inline MatchMetric match_metric() const { return match_metric_; }
  inline void SetMatchMetric(MatchMetric metric) { match_metric_ = metric; }
  
  inline int context_radius() const { return context_radius_; }
  inline void SetContextRadius(int radius) { context_radius_ = radius; }
  
  inline float min_initial_depth() const { return min_initial_depth_; }
  inline void SetMinInitialDepth(float depth) { min_initial_depth_ = depth; }
  
  inline float max_initial_depth() const { return max_initial_depth_; }
  inline void SetMaxInitialDepth(float depth) { max_initial_depth_ = depth; }
  
  inline int iteration_count() const { return iteration_count_; }
  inline void SetIterationCount(int count) { iteration_count_ = count; }
  
  inline float max_normal_2d_length() const { return max_normal_2d_length_; }
  inline void SetMaxNormal2DLength(float length) { max_normal_2d_length_ = length; }
  
  // Outlier filtering settings accessors
  inline float min_patch_variance() const { return min_patch_variance_; }
  inline void SetMinPatchVariance(float threshold) { min_patch_variance_ = threshold; }
  
  inline float cost_threshold_per_pixel() const { return cost_threshold_per_pixel_; }
  inline void SetCostThresholdPerPixel(float threshold) { cost_threshold_per_pixel_ = threshold; }
  
  inline int min_component_size() const { return min_component_size_; }
  inline void SetMinComponentSize(int size) { min_component_size_ = size; }
  
  inline float similar_depth_ratio() const { return similar_depth_ratio_; }
  inline void SetSimilarDepthRatio(float ratio) { similar_depth_ratio_ = ratio; }
  
  inline float required_range_min_depth() const { return required_range_min_depth_; }
  inline void SetRequiredRangeMinDepth(float depth) { required_range_min_depth_ = depth; }
  
  inline float required_range_max_depth() const { return required_range_max_depth_; }
  inline void SetRequiredRangeMaxDepth(float depth) { required_range_max_depth_ = depth; }
  
 private:
  template <class CameraT1, class CameraT2>
  void ComputeDepthMap_(
      const CameraT1& reference_camera,
      const Image<u8>& reference_image,
      const SE3f& reference_image_tr_global,
      const CameraT2& stereo_camera,
      const Image<u8>& stereo_image,
      const SE3f& stereo_image_tr_global,
      Image<float>* inv_depth_map);
  
  inline bool DepthIsSimilar(float inv_depth_1, float inv_depth_2) {
    float ratio = inv_depth_1 / inv_depth_2;
    if (ratio < 1) {
      ratio = 1.f / ratio;
    }
    return ratio < similar_depth_ratio_;
  }
  
  void RemoveSmallConnectedComponentsInInvDepthMap(
      float separator_value,
      int min_component_size,
      int min_x,
      int min_y,
      int max_x,
      int max_y,
      Image<float>* inv_depth_map);
  
  
  // PatchMatch stereo settings.
  // NOTE: min_initial_depth_ and max_initial_depth_ also provide the depth
  //       range for mutations. In addition, max_initial_depth_ is also used
  //       to filter depth estimates which are farther away.
  MatchMetric match_metric_ = MatchMetric::kZNCC;
  int context_radius_ = 3;
  float min_initial_depth_ = 0.1f;
  float max_initial_depth_ = 20.0f;
  int iteration_count_ = 20;
  float max_normal_2d_length_ = 0.8f;
  
  // Outlier filtering settings
  float min_patch_variance_ = 5 * 5;
  float cost_threshold_per_pixel_ = 10 * 10; // 3 * 3;
  int min_component_size_ = 5; //50;
  float similar_depth_ratio_ = 1.025f;
  float required_range_min_depth_ = 1.5f;
  float required_range_max_depth_ = 3.0f;
};

}
