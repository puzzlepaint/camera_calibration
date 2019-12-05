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

#include <curand_kernel.h>

#include "libvis/cuda/cuda_buffer.h"
#include "libvis/cuda/cuda_unprojection_lookup.h"
#include "libvis/cuda/pixel_corner_projector.h"
#include "libvis/image.h"
#include "libvis/libvis.h"
#include "libvis/sophus.h"

namespace vis {

// PatchMatch Stereo implementation using CUDA.
// 
// Notes about fronto-parallel staircasing artifacts:
// With SSD and ZNCC (and SAD), there is a fronto-parallel depth staircasing
// effect that seems to be caused by image noise (it can be simulated with noise
// on synthetic renderings).
// Using Census, no step artifacts are visible.
// --> It appears to be introduced by directly using the bilinearly filtered
//     pixel values. Idea: the places where the interpolation changes probably
//     often (but not always) cause extrema in the cost. Thus the depth is
//     likely to align with those places where the samples fall on pixel centers
//     in the stereo image.
// --> Possible fix: use bicubic filtering?
// --> Possible fix: better distribution of samples? For example, could try
//     a random pattern, or a pattern with half-pixel instead of one-pixel
//     distance between the samples.
// Furthermore, the angle threshold might prefer deleting points in-between
// steps which are close to the image borders, enhancing the effect’s visibility
// there.
class PatchMatchStereoCUDA {
 public:
  enum class MatchMetric {
    // Will not work well on images with differing lighting conditions or
    // specular lighting.
    kSSD = 0,
    
    // Better suited for most applications, however returns very noisy values
    // in homogeneous areas.
    kZNCC = 1,
    
    // Invariant against monotonic illumination changes
    kCensus = 2
  };
  
  PatchMatchStereoCUDA(int width, int height);
  
  /// Variant of ComputeDepthMap() for a single stereo image.
  /// TODO: Assumes that the mask is the same for the reference and stereo image.
  /// 
  /// The mask may be empty. Otherwise, pixels with value zero will be treated
  /// as being masked out.
  void ComputeDepthMap(
      const CUDAUnprojectionLookup2D& reference_unprojection,
      const Image<u8>& reference_image,
      const SE3f& reference_image_tr_global,
      const PixelCornerProjector& stereo_projection,
      const Image<u8>& mask,
      const Image<u8>& stereo_image,
      const SE3f& stereo_image_tr_global,
      Image<float>* inv_depth_map,
      Image<float>* lr_consistency_inv_depth_map = nullptr);
  
  /// Variant of ComputeDepthMap() for multiple stereo images.
  /// TODO: Assumes the same intrinsics and mask for all images.
  /// 
  /// The mask may be empty. Otherwise, pixels with value zero will be treated
  /// as being masked out.
  void ComputeDepthMap(
      const CUDAUnprojectionLookup2D& unprojection,
      const PixelCornerProjector& projection,
      const Image<u8>& mask,
      int num_images,
      const Image<u8>** images,
      const SE3f** images_tr_global,
      int reference_image_index,
      Image<float>* inv_depth_map);
  
  // Can be called after ComputeDepthMap() to get the estimated normal image.
  // A normal for a pixel is only meaningful if the corresponding depth value
  // is valid.
  void GetNormals(Image<Vec3f>* normals);
  
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
  
  inline int consistency_check_iteration_count() const { return consistency_check_iteration_count_; }
  inline void SetConsistencyCheckIterationCount(int count) { consistency_check_iteration_count_ = count; }
  
  inline float max_normal_2d_length() const { return max_normal_2d_length_; }
  inline void SetMaxNormal2DLength(float length) { max_normal_2d_length_ = length; }
  
  // Post processing settings accessors
  inline float bilateral_filter_sigma_xy() const { return bilateral_filter_sigma_xy_; }
  inline void SetBilateralFilterSigmaXY(float sigma_xy) { bilateral_filter_sigma_xy_ = sigma_xy; }
  
  inline float bilateral_filter_radius_factor() const { return bilateral_filter_radius_factor_; }
  inline void SetBilateralFilterRadiusFactor(float radius_factor) { bilateral_filter_radius_factor_ = radius_factor; }
  
  inline float bilateral_filter_sigma_inv_depth() const { return bilateral_filter_sigma_inv_depth_; }
  inline void SetBilateralFilterSigmaInvDepth(float sigma_inv_depth) { bilateral_filter_sigma_inv_depth_ = sigma_inv_depth; }
  
  // Outlier filtering settings accessors
  
  /// Maximum allowed depth value. Note that the value itself is actually the first value that is not allowed anymore, rather than the maximum.
  /// This allows to use infinity (which becomes zero in inverse depth).
  inline float max_depth() const { return max_depth_; }
  inline void SetMaxDepth(float max_depth) { max_depth_ = max_depth; }
  
  inline float min_epipolar_gradient_per_pixel() const { return min_epipolar_gradient_per_pixel_; }
  inline void SetMinEpipolarGradientPerPixel(float threshold) { min_epipolar_gradient_per_pixel_ = threshold; }
  
  inline float angle_threshold() const { return angle_threshold_; }
  inline void SetAngleThreshold(float threshold) { angle_threshold_ = threshold; }
  
  inline float cost_threshold() const { return cost_threshold_; }
  inline void SetCostThreshold(float threshold) { cost_threshold_ = threshold; }
  
  inline int min_component_size() const { return min_component_size_; }
  inline void SetMinComponentSize(int size) { min_component_size_ = size; }
  
  /// Similar depth ratio for determining whether two adjacent depth pixels can be in the same connected component, or not.
  inline float similar_depth_ratio() const { return similar_depth_ratio_; }
  inline void SetSimilarDepthRatio(float ratio) { similar_depth_ratio_ = ratio; }
  
  inline float required_range_min_depth() const { return required_range_min_depth_; }
  inline void SetRequiredRangeMinDepth(float depth) { required_range_min_depth_ = depth; }
  
  inline float required_range_max_depth() const { return required_range_max_depth_; }
  inline void SetRequiredRangeMaxDepth(float depth) { required_range_max_depth_ = depth; }
  
  inline float lr_consistency_factor_threshold() const { return lr_consistency_factor_threshold_; }
  inline void SetLRConsistencyFactorThreshold(float depth) { lr_consistency_factor_threshold_ = depth; }
  
  inline float lr_consistency_required_inlier_amount() const { return lr_consistency_required_inlier_amount_; }
  inline void SetLRConsistencyRequiredInlierAmount(float depth) { lr_consistency_required_inlier_amount_ = depth; }
  
  inline float second_best_min_distance_factor() const { return second_best_min_distance_factor_; }
  inline void SetSecondBestMinDistanceFactor(float factor) { second_best_min_distance_factor_ = factor; }
  
  inline float second_best_min_cost_factor() const { return second_best_min_cost_factor_; }
  inline void SetSecondBestMinCostFactor(float factor) { second_best_min_cost_factor_ = factor; }
  
 private:
  inline bool DepthIsSimilar(float inv_depth_1, float inv_depth_2) {
    float ratio = inv_depth_1 / inv_depth_2;
    if (ratio < 1) {
      ratio = 1.f / ratio;
    }
    return ratio < similar_depth_ratio_;
  }
  
  int ConvertMatchMetric(MatchMetric metric);
  
  void RemoveSmallConnectedComponentsInInvDepthMap(
      float separator_value,
      int min_component_size,
      int min_x,
      int min_y,
      int max_x,
      int max_y,
      Image<float>* inv_depth_map);
  
  void ComputeDepthMapImpl(
      int iterations,
      const CUDAUnprojectionLookup2D& unprojection,
      const PixelCornerProjector& projection,
      const Image<u8>& mask,
      int num_images,
      const Image<u8>** images,
      const vector<cudaTextureObject_t>& textures,
      const SE3f** images_tr_global,
      int reference_image_index,
      Image<float>* inv_depth_map,
      const vector<Image<float>>& consistency_inv_depth_maps = vector<Image<float>>());
  
  
  // PatchMatch stereo settings.
  // NOTE: min_initial_depth_ and max_initial_depth_ also provide the depth
  //       range for mutations. In addition, max_initial_depth_ is also used
  //       to filter depth estimates which are farther away.
  MatchMetric match_metric_ = MatchMetric::kZNCC;
  int context_radius_ = 3;
  float min_initial_depth_ = 0.3f;
  float max_initial_depth_ = 20.0f;
  int iteration_count_ = 100;
  int consistency_check_iteration_count_ = 50;
  float max_normal_2d_length_ = 1.0f;
  
  // Postprocessing settings
  float bilateral_filter_sigma_xy_ = 1.5f;
  float bilateral_filter_radius_factor_ = 2.0f;
  float bilateral_filter_sigma_inv_depth_ = 0.005f;
  
  // Outlier filtering settings
  float max_depth_ = 20.0f;
  float min_epipolar_gradient_per_pixel_ = 3;
  float angle_threshold_ = 3.141592f / 180.f * 80;  // 80 degrees
  float cost_threshold_ = 10 * 10; // 3 * 3;
  int min_component_size_ = 5; //50;
  float similar_depth_ratio_ = 1.025f;
  float required_range_min_depth_ = 2.0f;
  float required_range_max_depth_ = 3.0f;
  float lr_consistency_factor_threshold_ = 1.025f;
  float lr_consistency_required_inlier_amount_ = 0.75f;
  // Factor on the best result's depth for determining the range where the second best result cannot be
  float second_best_min_distance_factor_ = 1.025f;
  // Minimum factor on the best result's cost which the second best cost must be higher than for an unambiguous pixel.
  // Set to 1 to disable.
  float second_best_min_cost_factor_ = 1.4f;
  
  // CUDA memory
  CUDABufferPtr<u8> reference_image_gpu_;
  CUDABufferPtr<u8> stereo_image_gpu_;
  CUDABufferPtr<float> inv_depth_map_gpu_;
  CUDABufferPtr<float> inv_depth_map_gpu_2_;
  
  CUDABufferPtr<char2> normals_;
  CUDABufferPtr<float> costs_;
  CUDABufferPtr<float> costs_2_;
  CUDABufferPtr<curandState> random_states_;
  
  CUDABufferPtr<float> second_best_inv_depth_map_gpu_;
  CUDABufferPtr<char2> second_best_normals_;
  CUDABufferPtr<float> second_best_costs_;
  CUDABufferPtr<float> second_best_costs_2_;
};

}
