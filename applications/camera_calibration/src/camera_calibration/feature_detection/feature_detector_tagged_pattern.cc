// Copyright 2019 ETH Zürich, Thomas Schöps
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

#include "camera_calibration/feature_detection/feature_detector_tagged_pattern.h"

#include <unordered_map>
#include <unordered_set>

#include <apriltag.h>
#include <cuda_runtime.h>
#include <cub/util_type.cuh>
#include <libvis/dlt.h>
#include <libvis/cuda/cuda_buffer.h>
#include <libvis/image_display.h>
#include <libvis/timing.h>
#include <tag36h11.h>
#include <yaml-cpp/yaml.h>

#include "camera_calibration/feature_detection/cpu_refinement_by_matching.h"
#include "camera_calibration/feature_detection/cpu_refinement_by_symmetry.h"
#include "camera_calibration/feature_detection/cuda_refinement_by_matching.h"
#include "camera_calibration/feature_detection/cuda_refinement_by_symmetry.h"
#include "camera_calibration/feature_detection/cuda_util.cuh"
#include "camera_calibration/hash_vec2i.h"


namespace vis {

/// Attributes belonging to FeatureDetectorTaggedPattern that shall not be part
/// of its public interface (this e.g. avoids CUDA includes in the header).
struct FeatureDetectorTaggedPatternPrivate {
  FeatureDetectorTaggedPatternPrivate() = default;
  
  ~FeatureDetectorTaggedPatternPrivate() {
    if (cuda_objects_initialized) {
      cudaDestroyTextureObject(image_texture);
      cudaDestroyTextureObject(gradient_image_texture);
      cudaFree(cost_buffer);
    }
  }
  
  bool use_cuda = true;
  bool cuda_objects_initialized = false;
  
  CUDABufferPtr<u8> cuda_image;
  cudaTextureObject_t image_texture;
  
  CUDABufferPtr<float2> cuda_gradient_image;
  cudaTextureObject_t gradient_image_texture;
  
  int cost_buffer_size = 0;
  cub::KeyValuePair<int, float>* cost_buffer = nullptr;
  
  CUDABufferPtr<float2> position_temp_buffer;
  CUDABufferPtr<float4> state_buffer;
  CUDABufferPtr<float> rendered_samples;
  CUDABufferPtr<float> local_pattern_tr_pixel_buffer;
  
  vector<Vec2f> samples;  // homogeneously distributed in [-1, 1] x [-1, 1].
  CUDABufferPtr<float2> samples_gpu;
  CUDABufferPtr<float2> pattern_samples_gpu;
  
  /// Attention, the index of the object in this vector may be different from
  /// its pattern_index. Only the latter is relevant.
  vector<PatternData> patterns;
  
  apriltag_detector_t* apriltag_detector = nullptr;
  apriltag_family_t* apriltag_family = nullptr;
  
  int timing_output_counter = 0;
  unique_ptr<ImageDisplay> debug_display;
};


static bool CheckOppositeAngleAndLengthCriterion(
    const Vec2f& forward_feature_position,
    const Vec2f& middle_feature_position,
    const Vec2f& opposite_feature_position,
    bool debug) {
  Vec2f middle_to_forward = forward_feature_position - middle_feature_position;
  Vec2f middle_to_opposite = opposite_feature_position - middle_feature_position;
  
  constexpr float kOppositeLengthRatioThreshold = 1.5f;
  float middle_to_forward_length = middle_to_forward.norm();
  float middle_to_opposite_length = middle_to_opposite.norm();
  float ratio = std::max(middle_to_forward_length / middle_to_opposite_length,
                         middle_to_opposite_length / middle_to_forward_length);
  if (ratio > kOppositeLengthRatioThreshold) {
    if (debug) {
      LOG(WARNING) << "Rejection due to opposite length criterion (ratio: "
                   << ratio << ", threshold: " << kOppositeLengthRatioThreshold << ")";
    }
    return false;
  }
  
  constexpr float kOppositeAngleThreshold = 5;  // in degrees
  Vec2f middle_to_forward_direction = middle_to_forward / middle_to_forward_length;
  Vec2f middle_to_opposite_direction = middle_to_opposite / middle_to_opposite_length;
  if (middle_to_forward_direction.dot(middle_to_opposite_direction) >
      std::cos(M_PI / 180.f * (180 - kOppositeAngleThreshold))) {
    if (debug) {
      LOG(WARNING) << "Rejection due to opposite angle criterion (angle: "
                   << (180 - 180.f / M_PI * std::acos(middle_to_forward_direction.dot(middle_to_opposite_direction)))
                   << ", threshold: " << kOppositeAngleThreshold << ")";
    }
    return false;
  }
  
  return true;
}


FeatureDetectorTaggedPattern::FeatureDetectorTaggedPattern(
    const vector<string>& pattern_yaml_paths,
    int window_half_extent,
    FeatureRefinement refinement_type,
    bool use_cuda) {
  valid_ = true;
  d.reset(new FeatureDetectorTaggedPatternPrivate());
  d->use_cuda = use_cuda;
  this->window_half_extent = window_half_extent;
  this->refinement_type = refinement_type;
  
  SetPatternYAMLPaths(pattern_yaml_paths);
}

FeatureDetectorTaggedPattern::~FeatureDetectorTaggedPattern() {
  // required for unique_ptr with type that is incomplete in the header
}

bool FeatureDetectorTaggedPattern::SetPatternYAMLPaths(
      const vector<string>& paths) {
  this->pattern_yaml_paths = paths;
  
  d->patterns.clear();
  valid_ = true;
  
  bool is_first = true;
  for (const string& path : pattern_yaml_paths) {
    try {
      YAML::Node file_node = YAML::LoadFile(path);
      
      if (file_node.IsNull()) {
        LOG(ERROR) << "Cannot read file: " << path;
        valid_ = false;
        return false;
      }
      
      PatternData data;
      data.num_star_segments = file_node["num_star_segments"].as<int>();
      data.squares_x = file_node["squares_x"].as<int>();
      data.squares_y = file_node["squares_y"].as<int>();
      YAML::Node apriltags_node = file_node["apriltags"];
      data.tags.resize(apriltags_node.size());
      for (int i = 0; i < data.tags.size(); ++ i) {
        AprilTagInfo& tag = data.tags[i];
        YAML::Node tag_node = apriltags_node[i];
        
        tag.x = tag_node["tag_x"].as<int>();
        tag.y = tag_node["tag_y"].as<int>();
        tag.width = tag_node["width"].as<int>();
        tag.height = tag_node["height"].as<int>();
        tag.index = tag_node["index"].as<int>();
      }
      YAML::Node page_node = file_node["page"];
      data.page_width_mm = page_node["width_mm"].as<float>();
      data.page_height_mm = page_node["height_mm"].as<float>();
      data.pattern_start_x_mm = page_node["pattern_start_x_mm"].as<float>();
      data.pattern_start_y_mm = page_node["pattern_start_y_mm"].as<float>();
      data.pattern_end_x_mm = page_node["pattern_end_x_mm"].as<float>();
      data.pattern_end_y_mm = page_node["pattern_end_y_mm"].as<float>();
      d->patterns.push_back(data);
      
      if (is_first) {
        cell_length_in_meters = file_node["square_length_in_meters"].as<float>();
      } else {
        // TODO: Support this
        CHECK_EQ(cell_length_in_meters, file_node["square_length_in_meters"].as<float>()) << "Different square_length_in_meters settings for different patterns are not supported at the moment.";
      }
    } catch (const YAML::BadFile& ex) {
      LOG(ERROR) << "Cannot read pattern file: " << path;
      valid_ = false;
      return false;
    }
  }
  
  return true;
}

void FeatureDetectorTaggedPattern::DetectFeatures(
    const Image<Vec3u8>& image,
    vector<PointFeature>* features,
    Image<Vec3u8>* detection_visualization) {
  constexpr bool kDebug = false;
  constexpr bool kDebugStepByStep = false;
  
  Timer frame_timer("DetectFeatures()");
  
  // Prepare debug display.
  if (kDebug) {
    if (!d->debug_display) {
      d->debug_display.reset(new ImageDisplay());
    }
    d->debug_display->Clear();
    d->debug_display->Update(image, "Feature detection");
  }
  
  // Prepare visualization.
  if (detection_visualization) {
    detection_visualization->SetSize(image.size());
    detection_visualization->SetTo(image);
  }
  
  // Prepare sample positions.
  int max_sample_count = static_cast<int>(8.0 * (2 * window_half_extent + 1) * (2 * window_half_extent + 1) + 0.5);
  if (d->samples.empty() ||
      d->samples.size() < max_sample_count) {
    d->samples.resize(max_sample_count);
    srand(0);
    for (usize i = 0; i < d->samples.size(); ++ i) {
      d->samples[i] = Vec2f::Random();
    }
  }
  
  // Prepare CUDA resources.
  if (d->use_cuda) {
    PrepareCUDAResources(image.width(), image.height());
  }
  
  // Prepare AprilTag detector.
  PrepareAprilTagDetector();
  
  // Convert the image to grayscale.
  Image<u8> gray_image;
  image.ConvertToGrayscale(&gray_image);
  
  // Compute gradient image on CPU or GPU.
  Image<Vec2f> gradient_image;
  Image<float> gradmag_image;
  if (d->use_cuda) {
    d->cuda_image->UploadAsync(/*stream*/ 0, gray_image);
    ComputeGradientImageCUDA(d->cuda_image->ToCUDA(), gray_image.width(), gray_image.height(), &d->cuda_gradient_image->ToCUDA());
  } else {
    gradient_image.SetSize(gray_image.size());
    gradmag_image.SetSize(gray_image.size());
    int width = gray_image.width();
    int height = gray_image.height();
    for (int y = 0; y < height; ++ y) {
      for (int x = 0; x < width; ++ x) {
        int mx = std::max<int>(0, x - 1);
        int px = std::min<int>(width - 1, x + 1);
        
        int my = std::max<int>(0, y - 1);
        int py = std::min<int>(height - 1, y + 1);
        
        float dx = (gray_image(px, y) - static_cast<float>(gray_image(mx, y))) / (px - mx);
        float dy = (gray_image(x, py) - static_cast<float>(gray_image(x, my))) / (py - my);
        
        gradient_image(x, y) = Vec2f(dx, dy);
        gradmag_image(x, y) = gradient_image(x, y).norm();
      }
    }
  }
  
  // Detect AprilTags in the image. Downscale the image if it is too large,
  // since tag detection should not require high resolution.
  // TODO: Is the downsampling really useful?
  Timer apriltag_timer("AprilTag detection");
  constexpr int kMaxSideLengthForTagDetection = 1280;
  Image<u8>* tag_image = &gray_image;
  shared_ptr<Image<u8>> downscaled_image;
  shared_ptr<Image<u8>> downscaled_image_2;
  float tag_upscale_factor = 1;
  while ((tag_image->width() > kMaxSideLengthForTagDetection ||
          tag_image->height() > kMaxSideLengthForTagDetection) &&
         tag_image->width() % 2 == 0 && tag_image->height() % 2 == 0) {
    if (!downscaled_image) {
      downscaled_image.reset(new Image<u8>());
      downscaled_image_2.reset(new Image<u8>());
    }
    
    Image<u8>* dest_image = (tag_image == downscaled_image.get()) ? downscaled_image_2.get() : downscaled_image.get();
    tag_image->DownscaleToHalfSize(dest_image);
    tag_image = dest_image;
    tag_upscale_factor *= 2;
  }
  
  image_u8_t img_header = { .width = static_cast<int>(tag_image->width()),
      .height = static_cast<int>(tag_image->height()),
      .stride = static_cast<int>(tag_image->stride()),
      .buf = tag_image->data()
  };
  zarray_t* detections = apriltag_detector_detect(d->apriltag_detector, &img_header);
  apriltag_timer.Stop();
  
  static Vec3u8 debug_colors[8] = {
      Vec3u8(255, 80, 80),
      Vec3u8(255, 80, 255),
      Vec3u8(80, 255, 255),
      Vec3u8(0, 255, 0),
      Vec3u8(80, 80, 255),
      Vec3u8(127, 255, 127),
      Vec3u8(255, 160, 0),
      Vec3u8(255, 255, 0)};
  
  /// Mapping: feature_predictions[pattern_array_index][Vec2i(pattern_x, pattern_y)] -> feature prediction.
  vector<unordered_map<Vec2i, FeatureDetection>> feature_predictions;
  
  // Use the AprilTags to find initial feature predictions.
  PredictFeaturesNextToAprilTags(
      gray_image,
      detections,
      tag_upscale_factor,
      &feature_predictions,
      kDebug,
      debug_colors);
  apriltag_detections_destroy(detections);
  
  /// Mapping: feature_predictions[pattern_array_index][Vec2i(pattern_x, pattern_y)] -> final feature detection.
  vector<unordered_map<Vec2i, FeatureDetection>> feature_detections;
  
  /// Contains rejected detections: feature_rejections[pattern_array_index][Vec2i(pattern_x, pattern_y)]
  vector<unordered_set<Vec2i>> feature_rejections;
  
  // Iteratively extend the predicted and detected features.
  Timer predict_and_detect_timer("PredictAndDetectFeatures()");
  PredictAndDetectFeatures(
      gray_image,
      gradient_image,
      gradmag_image,
      &feature_predictions,
      &feature_detections,
      &feature_rejections,
      kDebug,
      kDebugStepByStep,
      debug_colors);
  predict_and_detect_timer.Stop();
  
  // Detect and remove outliers.
  for (auto& pattern_feature_detections : feature_detections) {
    // At least 5 detections are required for "0.75f * all_final_costs.size() + 0.5f",
    // which is used below, to be a valid index. Also, for this kind of outlier
    // detection, it is necessary to have a minimum number of detections to be
    // able to estimate a cost threshold for outliers.
    if (pattern_feature_detections.size() < 5) {
      pattern_feature_detections.clear();
      continue;
    }
    
    // Find outliers among the final feature refinement costs and discard the
    // associated features. Next to image borders, there is an increased chance
    // of outliers: since the grid search for feature refinement partly overlaps
    // the image border, it may not be able to jump out of local minima as it
    // can otherwise do (if the correct minimum is out of the image). So the
    // refined features may be local minima. Thus, we use a stricter threshold
    // in these places.
    // We also delete features which have less than 2 neighbors, since there is
    // less possibility for neighbor-based consistency checking for those.
    // NOTE: Since we are deleting features during the iteration over the
    //       features here, not all features will be treated equally (since
    //       for some of them, some neighbors might have already been deleted
    //       since the iteration went over them first, but for others, that may
    //       not be the case). However, this should not be an issue. One could
    //       run the procedure multiple times to get a more consistent result.
    vector<float> all_final_costs;
    all_final_costs.reserve(pattern_feature_detections.size());
    for (const auto& item : pattern_feature_detections) {
      all_final_costs.push_back(item.second.final_cost);
    }
    std::sort(all_final_costs.begin(), all_final_costs.end());
    
    // 1.5 is what matplotlib uses by default for outliers in box plots
    constexpr float outlier_removal_factor = 6.f;  // TODO: make configurable
    constexpr float outlier_removal_factor_near_borders = 1.f;  // TODO: make configurable
    float first_quartile_error = all_final_costs[0.25f * all_final_costs.size() + 0.5f];
    float third_quartile_error = all_final_costs[0.75f * all_final_costs.size() + 0.5f];
    float outlier_threshold = third_quartile_error + outlier_removal_factor * (third_quartile_error - first_quartile_error);
    float outlier_threshold_near_borders = third_quartile_error + outlier_removal_factor_near_borders * (third_quartile_error - first_quartile_error);
    if (kDebug) {
      LOG(INFO) << "Outlier threshold: " << outlier_threshold
                << " (first quartile: " << first_quartile_error
                << ", third quartile: " << third_quartile_error
                << ", max cost: " << all_final_costs.back() << ")";
    }
    
    // Loop over the features for possible deletion until nothing changes anymore
    bool something_changed = true;
    while (something_changed) {
      something_changed = false;
      
      for (auto it = pattern_feature_detections.begin(); it != pattern_feature_detections.end(); ) {
        bool delete_feature = false;
        
        // Delete feature due to outlier cost?
        if (!delete_feature) {
          bool is_near_border = it->second.position.x() < 2 * window_half_extent ||
                                it->second.position.y() < 2 * window_half_extent ||
                                it->second.position.x() > image.width() - 1 - 2 * window_half_extent ||
                                it->second.position.y() > image.height() - 1 - 2 * window_half_extent;
          if (it->second.final_cost > (is_near_border ? outlier_threshold_near_borders : outlier_threshold)) {
            delete_feature = true;
            
            if (kDebug) {
              // "Cross out" the feature
              constexpr float kCrossExtent = 10;
              d->debug_display->AddSubpixelLinePixelCornerConv(
                  it->second.position + Vec2f::Constant(0.5f) + Vec2f(kCrossExtent, kCrossExtent),
                  it->second.position + Vec2f::Constant(0.5f) + Vec2f(-kCrossExtent, -kCrossExtent),
                  Vec3u8(255, 0, 0));
              d->debug_display->AddSubpixelLinePixelCornerConv(
                  it->second.position + Vec2f::Constant(0.5f) + Vec2f(-kCrossExtent, kCrossExtent),
                  it->second.position + Vec2f::Constant(0.5f) + Vec2f(kCrossExtent, -kCrossExtent),
                  Vec3u8(255, 0, 0));
              
              if (kDebugStepByStep) {
                LOG(INFO) << "Debug: removed an outlier feature";
                d->debug_display->Update();
                std::getchar();
              }
            }
          }
        }
        
        // Delete the feature due to lack of neighbors?
        if (!delete_feature) {
          int num_neighbors = pattern_feature_detections.count(Vec2i(it->first.x() + 1, it->first.y())) +
                              pattern_feature_detections.count(Vec2i(it->first.x() - 1, it->first.y())) +
                              pattern_feature_detections.count(Vec2i(it->first.x(), it->first.y() + 1)) +
                              pattern_feature_detections.count(Vec2i(it->first.x(), it->first.y() - 1));
          delete_feature = (num_neighbors < 2);
          if (kDebug && delete_feature) {
            LOG(WARNING) << "Feature deleted due to lack of neighbors";
          }
        }
        
        // Delete feature if there is no direction to validate it, or if
        // validation from any valid direction fails.
        const int x_directions[] = {-1, 1,  0, 0};
        const int y_directions[] = { 0, 0, -1, 1};
        bool validated = false;
        for (int direction_index = 0; direction_index < 4; ++ direction_index) {
          Vec2i neighbor_coords(it->first.x() + x_directions[direction_index],
                                it->first.y() + y_directions[direction_index]);
          Vec2i neighbor_coords_2(it->first.x() + 2 * x_directions[direction_index],
                                  it->first.y() + 2 * y_directions[direction_index]);
          
          auto origin_it = pattern_feature_detections.find(neighbor_coords);
          auto opposite_it = pattern_feature_detections.find(neighbor_coords_2);
          if (origin_it != pattern_feature_detections.end() &&
              opposite_it != pattern_feature_detections.end()) {
            if (!CheckOppositeAngleAndLengthCriterion(
                it->second.position,
                origin_it->second.position,
                opposite_it->second.position,
                kDebug)) {
              // Since we do not know whether the current feature or the others
              // involved in the test are the problem, delete all of them.
              pattern_feature_detections.erase(origin_it);
              pattern_feature_detections.erase(opposite_it);
              
              delete_feature = true;
              if (kDebug) {
                LOG(WARNING) << "Features deleted due to CheckOppositeAngleAndLengthCriterion() fail";
              }
              break;
            }
            validated = true;
          }
        }
        if (!validated) {
          delete_feature = true;
          if (kDebug) {
            LOG(WARNING) << "Feature deleted since it could not be validated";
          }
        }
        
        // Delete feature if the neighbor directions are inconsistent with the
        // neighbor's neighbor directions.
        if (!delete_feature) {
          for (int direction_index = 0; direction_index < 4; ++ direction_index) {
            Vec2i neighbor_coords(it->first.x() + x_directions[direction_index],
                                  it->first.y() + y_directions[direction_index]);
            auto neighbor_it = pattern_feature_detections.find(neighbor_coords);
            if (neighbor_it == pattern_feature_detections.end()) {
              continue;
            }
            
            Vec2i perpendicular_dir(-y_directions[direction_index], x_directions[direction_index]);
            
            // Compute the perpendicular direction at the neighbor
            Vec2i perp_neighbor_1 = neighbor_coords + perpendicular_dir;
            Vec2i perp_neighbor_2 = neighbor_coords - perpendicular_dir;
            auto perp_neighbor_1_it = pattern_feature_detections.find(perp_neighbor_1);
            auto perp_neighbor_2_it = pattern_feature_detections.find(perp_neighbor_2);
            if (perp_neighbor_1_it == pattern_feature_detections.end() &&
                perp_neighbor_2_it == pattern_feature_detections.end()) {
              continue;
            }
            
            Vec2f neighbor_perp_direction = Vec2f::Zero();
            if (perp_neighbor_1_it != pattern_feature_detections.end()) {
              neighbor_perp_direction += perp_neighbor_1_it->second.position - neighbor_it->second.position;
            }
            if (perp_neighbor_2_it != pattern_feature_detections.end()) {
              neighbor_perp_direction -= perp_neighbor_2_it->second.position - neighbor_it->second.position;
            }
            neighbor_perp_direction.normalize();
            
            // Compute the perpendicular direction at the center
            Vec2i perp_center_1 = it->first + perpendicular_dir;
            Vec2i perp_center_2 = it->first - perpendicular_dir;
            auto perp_center_1_it = pattern_feature_detections.find(perp_center_1);
            auto perp_center_2_it = pattern_feature_detections.find(perp_center_2);
            if (perp_center_1_it == pattern_feature_detections.end() &&
                perp_center_2_it == pattern_feature_detections.end()) {
              continue;
            }
            
            Vec2f center_perp_direction = Vec2f::Zero();
            if (perp_center_1_it != pattern_feature_detections.end()) {
              center_perp_direction += perp_center_1_it->second.position - it->second.position;
            }
            if (perp_center_2_it != pattern_feature_detections.end()) {
              center_perp_direction -= perp_center_2_it->second.position - it->second.position;
            }
            center_perp_direction.normalize();
            
            // Make sure that the directions are similar
            constexpr float kNeighborDirectionAngleThreshold = 25;
            if (neighbor_perp_direction.dot(center_perp_direction) <
                std::cos(M_PI / 180.f * kNeighborDirectionAngleThreshold)) {
//               // DEBUG
//               constexpr float kCrossExtent = 10;
//               d->debug_display->AddSubpixelLinePixelCornerConv(
//                   it->second.position + Vec2f::Constant(0.5f) + Vec2f(kCrossExtent, kCrossExtent),
//                   it->second.position + Vec2f::Constant(0.5f) + Vec2f(-kCrossExtent, -kCrossExtent),
//                   Vec3u8(255, 0, 0));
//               d->debug_display->AddSubpixelLinePixelCornerConv(
//                   it->second.position + Vec2f::Constant(0.5f) + Vec2f(-kCrossExtent, kCrossExtent),
//                   it->second.position + Vec2f::Constant(0.5f) + Vec2f(kCrossExtent, -kCrossExtent),
//                   Vec3u8(255, 0, 0));
//               
//               if (perp_center_1_it != pattern_feature_detections.end()) {
//                 d->debug_display->AddSubpixelLinePixelCornerConv(
//                     it->second.position + Vec2f::Constant(0.5f),
//                     perp_center_1_it->second.position + Vec2f::Constant(0.5f),
//                     Vec3u8(0, 0, 255));
//               }
//               if (perp_center_2_it != pattern_feature_detections.end()) {
//                 d->debug_display->AddSubpixelLinePixelCornerConv(
//                     it->second.position + Vec2f::Constant(0.5f),
//                     perp_center_2_it->second.position + Vec2f::Constant(0.5f),
//                     Vec3u8(0, 0, 255));
//               }
//               
//               if (perp_neighbor_1_it != pattern_feature_detections.end()) {
//                 d->debug_display->AddSubpixelLinePixelCornerConv(
//                     neighbor_it->second.position + Vec2f::Constant(0.5f),
//                     perp_neighbor_1_it->second.position + Vec2f::Constant(0.5f),
//                     Vec3u8(0, 0, 255));
//               }
//               if (perp_neighbor_2_it != pattern_feature_detections.end()) {
//                 d->debug_display->AddSubpixelLinePixelCornerConv(
//                     neighbor_it->second.position + Vec2f::Constant(0.5f),
//                     perp_neighbor_2_it->second.position + Vec2f::Constant(0.5f),
//                     Vec3u8(0, 0, 255));
//               }
//               
// //               if (kDebugStepByStep) {
//                 LOG(INFO) << "Debug: removed an outlier feature due to a neighbor direction inconsistency ("
//                           << ((180.f / M_PI) * std::acos(neighbor_perp_direction.dot(center_perp_direction))) << "degrees)";
//                 d->debug_display->Update();
//                 std::getchar();
// //               }
              
              delete_feature = true;
              if (kDebug) {
                LOG(WARNING) << "Feature deleted due to inconsistent neighbor directions";
              }
              break;
            }
          }
        }
        
        // Perform the deletion if necessary and go to the next element.
        if (delete_feature) {
          it = pattern_feature_detections.erase(it);
          something_changed = true;
        } else {
          ++ it;
        }
      }  // loop over detected features in pattern
    }  // loop over feature deletion iterations
  }  // loop over patterns
  
  // Output the detection results.
  int pattern_feature_base_index = 0;
  for (usize pattern_array_index = 0; pattern_array_index < d->patterns.size(); ++ pattern_array_index) {
    const PatternData& pattern = d->patterns[pattern_array_index];
    auto& pattern_feature_detections = feature_detections.at(pattern_array_index);
    
    constexpr int kMinX = 0;
    constexpr int kMinY = 0;
    const int kMaxX = pattern.squares_x - 2;
    const int kMaxY = pattern.squares_y - 2;
    
    // Number of features in the calibration target.
    const int kNumCorners = pattern.ComputeFeatureCount();
    
    int output_index = pattern_feature_base_index;
    pattern_feature_base_index += kNumCorners;
    
    features->reserve(features->size() + kNumCorners);
    for (int y = kMinY; y <= kMaxY; ++ y) {
      for (int x = kMinX; x <= kMaxX; ++ x) {
        if (!pattern.IsValidFeatureCoord(x, y)) {
          continue;
        }
        
        auto it = pattern_feature_detections.find(Vec2i(x, y));
        if (it != pattern_feature_detections.end()) {
          features->emplace_back(it->second.position + Vec2f::Constant(0.5f), output_index);
        }
        ++ output_index;
      }
    }
  }
  
  // Create the detection visualization.
  if (detection_visualization) {
    for (usize i = 0; i < feature_detections.size(); ++ i) {
      const Vec3u8& debug_color = debug_colors[d->patterns[i].tags[0].index % 8];
      auto& pattern_feature_detections = feature_detections[i];
      
      for (auto it = pattern_feature_detections.cbegin(); it != pattern_feature_detections.cend(); ++ it) {
        // Highlight feature location by drawing a square on it
        Vec2i int_position = (it->second.position + Vec2f::Constant(0.5f)).cast<int>();
        constexpr int kRadius = 2;
        const int min_x = int_position.x() - kRadius;
        const int max_x = int_position.x() + kRadius;
        const int min_y = int_position.y() - kRadius;
        const int max_y = int_position.y() + kRadius;
        for (int y = min_y + 1; y <= max_y - 1; ++ y) {
          if (y < 0 || y >= detection_visualization->height()) {
            continue;
          }
          if (min_x >= 0) {
            detection_visualization->at(min_x, y) = debug_color;
          }
          if (max_x >= 0) {
            detection_visualization->at(max_x, y) = debug_color;
          }
        }
        for (int x = min_x; x <= max_x; ++ x) {
          if (x < 0 || x >= detection_visualization->width()) {
            continue;
          }
          if (min_y >= 0) {
            detection_visualization->at(x, min_y) = debug_color;
          }
          if (max_y >= 0) {
            detection_visualization->at(x, max_y) = debug_color;
          }
        }
        
        // Line to x+1
        Vec2i plus_x_coord = it->first + Vec2i(1, 0);
        auto plus_x_it = pattern_feature_detections.find(plus_x_coord);
        if (plus_x_it != pattern_feature_detections.end()) {
          Vec2i plus_x_int_position = (plus_x_it->second.position + Vec2f::Constant(0.5f)).cast<int>();
          plus_x_int_position = plus_x_int_position.cwiseMax(Vec2i(0, 0)).cwiseMin(detection_visualization->size().cast<int>() - Vec2i(1, 1));
          detection_visualization->DrawLine(
              int_position.x(), int_position.y(),
              plus_x_int_position.x(), plus_x_int_position.y(),
              debug_color);
        }
        
        // Line to y+1
        Vec2i plus_y_coord = it->first + Vec2i(0, 1);
        auto plus_y_it = pattern_feature_detections.find(plus_y_coord);
        if (plus_y_it != pattern_feature_detections.end()) {
          Vec2i plus_y_int_position = (plus_y_it->second.position + Vec2f::Constant(0.5f)).cast<int>();
          plus_y_int_position = plus_y_int_position.cwiseMax(Vec2i(0, 0)).cwiseMin(detection_visualization->size().cast<int>() - Vec2i(1, 1));
          detection_visualization->DrawLine(
              int_position.x(), int_position.y(),
              plus_y_int_position.x(), plus_y_int_position.y(),
              debug_color);
        }
      }
    }
  }
  
  frame_timer.Stop();
  
  // Debug: display the detected tags and features.
  if (kDebug) {
    d->debug_display->Update(image, "Feature detection");
    std::getchar();
  }
  
  // Output timings in regular intervals
  ++ d->timing_output_counter;
  if (d->timing_output_counter % 10 == 0) {
    LOG(INFO) << Timing::print(kSortByTotal);
  }
}

int FeatureDetectorTaggedPattern::GetPatternCount() const {
  return d->patterns.size();
}

const PatternData& FeatureDetectorTaggedPattern::GetPatternData(int pattern_index) const {
  return d->patterns[pattern_index];
}

void FeatureDetectorTaggedPattern::GetCorners(
    int pattern_index,
    unordered_map<int, Vec2i>* feature_id_to_coord) const {
  // NOTE: Inefficient implementation. Should cache the "start index" for each
  //       pattern instead of determining it anew in each call to this function.
  int feature_index = 0;
  
  for (usize p = 0; p < d->patterns.size(); ++ p) {
    const PatternData& pattern = d->patterns[p];
    
    for (int y = 0; y <= pattern.squares_y - 2; ++ y) {
      for (int x = 0; x <= pattern.squares_x - 2; ++ x) {
        if (!pattern.IsValidFeatureCoord(x, y)) {
          continue;
        }
        
        if (p == pattern_index) {
          feature_id_to_coord->insert(make_pair(feature_index, Vec2i(x, y)));
        }
        
        ++ feature_index;
      }
    }
    
    if (p == pattern_index) {
      break;
    }
  }
}

void FeatureDetectorTaggedPattern::PredictFeaturesNextToAprilTags(
    const Image<u8>& image,
    zarray* detections,
    float tag_upscale_factor,
    vector<unordered_map<Vec2i, FeatureDetection>>* feature_predictions,
    bool debug,
    Vec3u8 debug_colors[8]) {
  feature_predictions->resize(d->patterns.size());
  
  for (int detection_index = 0; detection_index < zarray_size(detections); ++ detection_index) {
    // Get the AprilTag detection
    apriltag_detection_t* tag_detection;
    zarray_get(detections, detection_index, &tag_detection);
    if (tag_detection->hamming > 0) {
      continue;
    }
    
    // Discard the detection in case one of the corners is outside of the image.
    // It happened that the tag detection was inaccurate in these cases, causing
    // inconsistencies in the feature detections that lead to rejecting the
    // whole pattern detection.
    bool discard = false;
    for (int i = 0; i < 4; ++ i) {
      Vec2f tag_corner = Vec2f(tag_upscale_factor * tag_detection->p[i][0],
                               tag_upscale_factor * tag_detection->p[i][1]);
      if (!image.ContainsPixelCenterConv(tag_corner - Vec2f::Constant(0.5f))) {
        discard = true;
        break;
      }
    }
    if (discard) {
      continue;
    }
    
    // Find the pattern that this tag belongs to
    int pattern_array_index = -1;
    const AprilTagInfo* tag_info = nullptr;
    for (usize i = 0; i < d->patterns.size(); ++ i) {
      for (AprilTagInfo& tag : d->patterns[i].tags) {
        if (tag.index == tag_detection->id) {
          pattern_array_index = i;
          tag_info = &tag;
          break;
        }
      }
      if (pattern_array_index >= 0) {
        break;
      }
    }
    if (pattern_array_index < 0) {
      LOG(WARNING) << "Detected AprilTag " << tag_detection->id << " which does not belong to a known pattern.";
      continue;
    }
    
    const PatternData& pattern = d->patterns[pattern_array_index];
    const Vec3u8& debug_color = debug_colors[pattern.tags[0].index % 8];
    
    // Debug: draw tag outlines.
    if (debug) {
      for (int c = 0; c < 4; ++ c) {
        int k = (c + 1) % 4;
        d->debug_display->AddSubpixelLinePixelCornerConv(
            tag_upscale_factor * Vec2f(tag_detection->p[c][0], tag_detection->p[c][1]),
            tag_upscale_factor * Vec2f(tag_detection->p[k][0], tag_detection->p[k][1]),
            debug_color);
      }
    }
    
    // Convert tag homography to Mat3f
    Mat3f tag_H;
    for (int row = 0; row < 3; ++ row) {
      for (int col = 0; col < 3; ++ col) {
        tag_H(row, col) = MATD_EL(tag_detection->H, row, col);
      }
    }
    tag_H.topRows<2>() *= tag_upscale_factor;
    
    // Compute and store the feature predictions for features next to this tag
    const int num_seeds = 2 * (tag_info->width + 1 + tag_info->height + 1);
    
    // Tag coordinates go from (-1, -1) to (1, 1) within the square detected by
    // the AprilTag library (i.e., the sqare with black inside and white
    // outside).
    const float square_length_in_tag_coords_x = 2 * (10 / 8.f) / tag_info->width;
    const float square_length_in_tag_coords_y = 2 * (10 / 8.f) / tag_info->height;
    
    // Insert seed points by extending the AprilTag edges.
    for (int c = 0; c < num_seeds; ++ c) {
      Vec2f predicted_corner_coords_float;
      Vec2i corner_coord;
      int side = c / (2 * (tag_info->width + 1));
      if (side == 0) {
        // Up- or downwards
        int sign = (c < tag_info->width + 1) ? 1 : -1;
        int i = (sign == 1) ?
                c :
                (c - (tag_info->width + 1));
        
        predicted_corner_coords_float = Vec2f(
            -1 - 1 / 4.f + i * square_length_in_tag_coords_x,
            sign * (1 + 1 / 4.f + square_length_in_tag_coords_y));
        corner_coord = Vec2i(
            tag_info->x - 1 + i,
            tag_info->y - 2 + ((sign == 1) ? (tag_info->height + 2) : 0));
      } else {  // side == 1
        // To the left or right
        int sign = ((c - 2 * (tag_info->width + 1)) < tag_info->height + 1) ? 1 : -1;
        int i = (sign == 1) ?
                (c - 2 * (tag_info->width + 1)) :
                (c - 2 * (tag_info->width + 1) - (tag_info->height + 1));
        
        predicted_corner_coords_float = Vec2f(
            sign * (1 + 1 / 4.f + square_length_in_tag_coords_x),
            -1 - 1 / 4.f + i * square_length_in_tag_coords_y);
        corner_coord = Vec2i(
            tag_info->x - 2 + ((sign == 1) ? (tag_info->width + 2) : 0),
            tag_info->y - 1 + i);
      }
      
      if (!pattern.IsValidFeatureCoord(corner_coord.x(), corner_coord.y())) {
        continue;
      }
      
      // Original coords are in pixel corner convention (at least I think so, not completely sure).
      Vec3f predicted_position_h = tag_H * predicted_corner_coords_float.homogeneous();
      Vec2f predicted_position = predicted_position_h.hnormalized() - Vec2f::Constant(0.5f);  // in pixel center convention
      
      // Check if any neighbor of this feature is predicted to be closer than a
      // minimum feature distance. In this case, do not add the feature (since
      // it will probably not be possible to detect it properly).
      const int x_directions[] = {-1, 1,  0, 0};
      const int y_directions[] = { 0, 0, -1, 1};
      bool reject_seed = false;
      for (int direction_index = 0; direction_index < 4; ++ direction_index) {
        Vec2f neighbor_coords(predicted_corner_coords_float.x() + square_length_in_tag_coords_x * x_directions[direction_index],
                              predicted_corner_coords_float.y() + square_length_in_tag_coords_y * y_directions[direction_index]);
        
        Vec3f predicted_neighbor_position_h = tag_H * neighbor_coords.homogeneous();
        Vec2f predicted_neighbor_position = predicted_neighbor_position_h.hnormalized() - Vec2f::Constant(0.5f);  // in pixel center convention
        
        float squared_distance = (predicted_neighbor_position - predicted_position).squaredNorm();
        constexpr float kMinimumStartingFeatureDistance = 2 * 5;  // in pixels; TODO: make configurable
        if (!(squared_distance >= kMinimumStartingFeatureDistance * kMinimumStartingFeatureDistance)) {
          reject_seed = true;
          break;
        }
      }
      
      if (!reject_seed) {
        FeatureDetection new_feature;
        new_feature.position = predicted_position;
        new_feature.pattern_coordinate = Vec2i(corner_coord.x(), corner_coord.y());
        new_feature.extension_direction = Vec2i::Zero();
        new_feature.final_cost = -1;
        
        // Compute homography with local coordinate systems
        Mat3f to_local_pattern;
        to_local_pattern << square_length_in_tag_coords_x,                             0, predicted_corner_coords_float.x(),
                            0,                             square_length_in_tag_coords_y, predicted_corner_coords_float.y(),
                            0,                                                         0,                                 1;
        Mat3f to_local_image;
        to_local_image << 1, 0, -predicted_position.x() - 0.5f,
                          0, 1, -predicted_position.y() - 0.5f,
                          0, 0,                              1;
        new_feature.local_pixel_tr_pattern = to_local_image * tag_H * to_local_pattern;
        
        feature_predictions->at(pattern_array_index).insert(make_pair(
            new_feature.pattern_coordinate,
            new_feature));
      }
    }
  }
}

/// Helper struct for finding the best points to compute a homography from.
struct Match {
  inline bool operator< (const Match& other) const {
    return squared_distance_to_new_coords < other.squared_distance_to_new_coords;
  }
  
  /// In pixel-center convention
  Vec2f pixel_coords;
  
  /// In square units; may be sub-square (for AprilTag corners)
  Vec2f pattern_coords;
  
  float squared_distance_to_new_coords;
};

void FeatureDetectorTaggedPattern::PredictAndDetectFeatures(
    const Image<u8>& image,
    const Image<Vec2f>& gradient_image,
    const Image<float>& gradmag_image,
    vector<unordered_map<Vec2i, FeatureDetection>>* feature_predictions,
    vector<unordered_map<Vec2i, FeatureDetection>>* feature_detections,
    vector<unordered_set<Vec2i>>* feature_rejections,
    bool debug,
    bool debug_step_by_step,
    Vec3u8 debug_colors[8]) {
  const int kIncrementalPredictionErrorThreshold = window_half_extent * 4 / 5.f;  // In pixels. TODO: make configurable
  constexpr float kMinimumFeatureDistance = 5;  // in pixels; TODO: make configurable
  
  int num_predictions = 0;
  for (auto& pattern_feature_predictions : *feature_predictions) {
    num_predictions += pattern_feature_predictions.size();
  }
  
  vector<bool> discard_pattern_detection(feature_predictions->size(), false);
  feature_detections->resize(feature_predictions->size());
  feature_rejections->resize(feature_predictions->size());
  
  while (num_predictions > 0) {
    if (debug) {
      // Show all feature predictions in gray
      for (usize pattern_array_index = 0; pattern_array_index < feature_predictions->size(); ++ pattern_array_index) {
        auto& pattern_feature_predictions = feature_predictions->at(pattern_array_index);
        
        for (auto& item : pattern_feature_predictions) {
          d->debug_display->AddSubpixelDotPixelCornerConv(
              item.second.position + Vec2f::Constant(0.5f),
              Vec3u8(127, 127, 127));
        }
      }
      if (debug_step_by_step) {
        LOG(INFO) << "Showing new predictions (and neighbor validations)";
        d->debug_display->Update();
        std::getchar();
      }
    }
    
    // Refine all feature predictions and convert them to detected features if
    // the refinement was successful. Submit all refinement requests at the same
    // time, such that they can be performed in parallel.
    vector<Vec3i> new_detections;  // format: (pattern_array_index, pattern_x, pattern_y).
    
    vector<FeatureDetection> features_for_refinement;
    features_for_refinement.reserve(128);
    vector<FeatureDetection> refined_detections;
    refined_detections.reserve(128);
    for (usize pattern_array_index = 0; pattern_array_index < feature_predictions->size(); ++ pattern_array_index) {
      auto& pattern_feature_predictions = feature_predictions->at(pattern_array_index);
      for (auto& item : pattern_feature_predictions) {
        features_for_refinement.push_back(item.second);
        refined_detections.emplace_back();
      }
    }
    
    Timer refine_feature_detections_timer("RefineFeatureDetections()");
    RefineFeatureDetections(
        image,
        gradient_image,
        gradmag_image,
        features_for_refinement.size(),
        features_for_refinement.data(),
        refined_detections.data(),
        debug,
        debug_step_by_step);
    refine_feature_detections_timer.Stop();
    
    int index = 0;
    for (usize pattern_array_index = 0; pattern_array_index < feature_predictions->size(); ++ pattern_array_index) {
      const PatternData& pattern = d->patterns[pattern_array_index];
      auto& pattern_feature_predictions = feature_predictions->at(pattern_array_index);
      auto& pattern_feature_detections = feature_detections->at(pattern_array_index);
      
      for (auto& item : pattern_feature_predictions) {
        const FeatureDetection& predicted_feature = item.second;
        FeatureDetection& refined_feature = refined_detections[index];
        ++ index;
        
        // For features discarded during refinement, the cost is set to a negative value.
        if (refined_feature.final_cost < 0) {
          continue;
        }
        
        // Check whether the returned position is within a reasonable range of the prediction.
        if (!((refined_feature.position - predicted_feature.position).squaredNorm() <=
              kIncrementalPredictionErrorThreshold * kIncrementalPredictionErrorThreshold)) {
          continue;
        }
        
        // Check whether the returned position is too close to an existing one.
        bool reject_detection = false;
        for (auto& existing_detection : pattern_feature_detections) {
          float squared_distance = (existing_detection.second.position - refined_feature.position).squaredNorm();
          if (!(squared_distance >= kMinimumFeatureDistance * kMinimumFeatureDistance)) {
            reject_detection = true;
            break;
          }
        }
        if (reject_detection) {
          continue;
        }
        
        refined_feature.pattern_coordinate = predicted_feature.pattern_coordinate;
        refined_feature.local_pixel_tr_pattern = predicted_feature.local_pixel_tr_pattern;
        // refined_feature.num_validations = 0;
        
        // // Check whether, if going back the direction this extension was done,
        // // the angles to this new feature and the feature opposite to it are
        // // consistent. Also, the lengths should be somewhat similar.
        // if (predicted_feature.extension_direction != Vec2i::Zero()) {
        //   Vec2i origin_detection = item.first - predicted_feature.extension_direction;
        //   auto origin_it = pattern_feature_detections.find(origin_detection);
        //   Vec2i opposite_detection = item.first - 2 * predicted_feature.extension_direction;
        //   auto opposite_it = pattern_feature_detections.find(opposite_detection);
        //   if (origin_it != pattern_feature_detections.end() &&
        //       opposite_it != pattern_feature_detections.end()) {
        //     if (!CheckOppositeAngleAndLengthCriterion(
        //         refined_feature.position,
        //         origin_it->second.position,
        //         opposite_it->second.position,
        //         debug)) {
        //       continue;
        //     }
        //     ++ refined_feature.num_validations;
        //   }
        // }
        
        // Add the refined position as a new detection.
        pattern_feature_detections.insert(make_pair(item.first, refined_feature));
        new_detections.push_back(Vec3i(pattern_array_index, item.first.x(), item.first.y()));
        
        if (debug) {
          d->debug_display->AddSubpixelDotPixelCornerConv(
              refined_feature.position + Vec2f::Constant(0.5f),
              debug_colors[pattern.tags[0].index % 8]);
        }
      }
    }
    
    for (auto& pattern_feature_predictions : *feature_predictions) {
      pattern_feature_predictions.clear();
    }
    num_predictions = 0;
    
    if (debug && debug_step_by_step) {
      LOG(INFO) << "Showing new refined detections";
      d->debug_display->Update();
      std::getchar();
    }
    
    // Iterate over all potential feature positions next to newly detected ones.
    // For each position, compute a homography using at least 4 of the closest
    // existing feature detections of that pattern (or more in case those are
    // collinear) to predict its image position.
    for (const Vec3i& new_detection : new_detections) {
      int pattern_array_index = new_detection.x();
      if (discard_pattern_detection[pattern_array_index]) {
        continue;
      }
      const PatternData& pattern = d->patterns[pattern_array_index];
      auto& pattern_feature_predictions = feature_predictions->at(pattern_array_index);
      auto& pattern_feature_detections = feature_detections->at(pattern_array_index);
      auto& pattern_feature_rejections = feature_rejections->at(pattern_array_index);
      
      if (pattern_feature_rejections.count(Vec2i(new_detection.y(), new_detection.z()))) {
        continue;
      }
      FeatureDetection& feature_detection = pattern_feature_detections.at(Vec2i(new_detection.y(), new_detection.z()));
      
      const int x_directions[] = {-1, 1,  0, 0};
      const int y_directions[] = { 0, 0, -1, 1};
      
      for (int direction_index = 0; direction_index < 4; ++ direction_index) {
        Vec2i neighbor_coords(new_detection.y() + x_directions[direction_index],
                              new_detection.z() + y_directions[direction_index]);
        if (!pattern.IsValidFeatureCoord(neighbor_coords.x(), neighbor_coords.y())) {
          continue;
        }
        
        // If this neighbor point was already predicted, there is nothing more
        // to do at the moment.
        if (pattern_feature_predictions.count(neighbor_coords) > 0) {
          continue;
        }
        
        // Compute the homography based on the closest detected points
        // (including AprilTag detections).
        vector<Match> matches;
        matches.reserve(
            pattern_feature_detections.size() +  // features
            0 /*4 * 2 * pattern.apriltag_length_in_squares*/);  // AprilTag corners
        
        // TODO: The matches arising from the AprilTag corners should be added here as well!
        
        for (auto& item : pattern_feature_detections) {
          if (item.first == neighbor_coords) {
            // For verifying detected features, do not use themselves for
            // homography estimation.
            continue;
          }
          matches.emplace_back();
          Match& new_match = matches.back();
          new_match.pattern_coords = item.first.cast<float>();
          new_match.pixel_coords = item.second.position;
          new_match.squared_distance_to_new_coords = (new_match.pattern_coords - neighbor_coords.cast<float>()).squaredNorm();
        }
        
        if (matches.size() < 4) {
          // Not enough points to estimate a homography.
          continue;
        }
        
        // TODO: Instead of searching over all matches and sorting them here,
        //       since the points are on a regular grid, it would be possible to
        //       iterate over possible neighboring points directly in
        //       (pre-computed) sorted order. This way, only the most relevant
        //       neighbors will be looked at.
        int last_match = 3;
        int sorted_until = std::min<int>(last_match + 3, matches.size() - 1);
        std::partial_sort(matches.begin(), matches.begin() + sorted_until + 1, matches.end());
        
        bool is_degenerate = true;
        while (true) {
          // Are the matches in a non-degenerate configuration? For a valid
          // configuration, if fitting a line going through as many of the
          // points as possible, at least two points must not lie on that line.
          // We can test this by checking against two lines going through
          // different points, since if the configuration is invalid, at least
          // one of these lines corresponds to the fitted line mentioned above.
          Vec2f line_0_1_direction = (matches[1].pattern_coords - matches[0].pattern_coords).normalized();
          Vec2f line_2_3_direction = (matches[3].pattern_coords - matches[2].pattern_coords).normalized();
          int num_not_on_0_1 = 0;
          int num_not_on_2_3 = 0;
          for (int i = 0; i <= last_match; ++ i) {
            if (i > 1) {
              Vec2f direction_from_0 = (matches[i].pattern_coords - matches[0].pattern_coords).normalized();
              if (fabs(line_0_1_direction.dot(direction_from_0)) < 0.95) {
                ++ num_not_on_0_1;
              }
            }
            
            if (i < 2 || i > 3) {
              Vec2f direction_from_2 = (matches[i].pattern_coords - matches[2].pattern_coords).normalized();
              if (fabs(line_2_3_direction.dot(direction_from_2)) < 0.95) {
                ++ num_not_on_2_3;
              }
            }
          }
          if (num_not_on_0_1 >= 2 && num_not_on_2_3 >= 2) {
            is_degenerate = false;
            break;
          }
          
          // The configuration is degenerate. Can we add another point?
          ++ last_match;
          if (last_match >= matches.size()) {
            break;
          } else if (last_match > sorted_until) {
            std::sort(matches.begin() + sorted_until, matches.end());
            sorted_until = matches.size();
          }
        }
        if (is_degenerate) {
          // Even adding all available points did not result in a non-degenerate
          // configuration.
          continue;
        }
        
        vector<Vec2f> pattern_coords(last_match + 1);
        vector<Vec2f> pixel_coords(last_match + 1);
        for (int i = 0; i <= last_match; ++ i) {
          pattern_coords[i] = matches[i].pattern_coords;
          pixel_coords[i] = matches[i].pixel_coords;
        }
        Mat3f homography = NormalizedDLT(pattern_coords.data(), pixel_coords.data(), last_match + 1);
        
        // Predict the neighbor's position.
        Vec2f predicted_position = Vec3f(homography * neighbor_coords.cast<float>().homogeneous()).hnormalized();
        
        auto neighbor_it = pattern_feature_detections.find(neighbor_coords);
        if (neighbor_it == pattern_feature_detections.end()) {
          // This neighbor point was not detected yet.
          // Add it as a new feature prediction if it was not rejected before.
          if (pattern_feature_rejections.count(neighbor_coords) == 0) {
            FeatureDetection new_feature;
            new_feature.position = predicted_position;
            new_feature.pattern_coordinate = neighbor_coords;
            new_feature.extension_direction = Vec2i(x_directions[direction_index], y_directions[direction_index]);
            new_feature.final_cost = -1;
            
            // Compute homography with local coordinate systems
            Mat3f to_local_pattern;
            to_local_pattern << 1, 0, neighbor_coords.x(),
                                0, 1, neighbor_coords.y(),
                                0, 0,                   1;
            Mat3f to_local_image;
            to_local_image << 1, 0, -predicted_position.x(),
                              0, 1, -predicted_position.y(),
                              0, 0,                    1;
            new_feature.local_pixel_tr_pattern = to_local_image * homography * to_local_pattern;
            
            feature_predictions->at(pattern_array_index).insert(make_pair(
                new_feature.pattern_coordinate,
                new_feature));
            
            ++ num_predictions;
          }
        } else {
          // This neighbor point was already detected. Verify that we estimate
          // it in roughly the same location (otherwise, something might be wrong).
          if (!((neighbor_it->second.position - predicted_position).squaredNorm() <=
                kIncrementalPredictionErrorThreshold * kIncrementalPredictionErrorThreshold)) {
            if (debug) {
              LOG(WARNING) << "Corner position inconsistency detected when verifying an existing corner. Predicted position: white dot. Previous position: red dot (connected by red line). Matches used for homography estimation (count: " << (last_match + 1) << ") framed in white.";
              
              d->debug_display->AddSubpixelDotPixelCornerConv(
                  predicted_position + Vec2f::Constant(0.5f),
                  Vec3u8(255, 255, 255));
              d->debug_display->AddSubpixelLinePixelCornerConv(
                  feature_detection.position + Vec2f::Constant(0.5f),
                  predicted_position + Vec2f::Constant(0.5f),
                  Vec3u8(255, 255, 255));
              
              d->debug_display->AddSubpixelDotPixelCornerConv(
                  neighbor_it->second.position + Vec2f::Constant(0.5f),
                  Vec3u8(255, 0, 0));
              d->debug_display->AddSubpixelLinePixelCornerConv(
                  neighbor_it->second.position + Vec2f::Constant(0.5f),
                  predicted_position + Vec2f::Constant(0.5f),
                  Vec3u8(255, 0, 0));
              
              for (int i = 0; i <= last_match; ++ i) {
                LOG(1) << "  Match pixel position: " << matches[i].pixel_coords.transpose() << ", pattern position: " << matches[i].pattern_coords.transpose();
                
                constexpr float kFramingExtent = 5;
                d->debug_display->AddSubpixelLinePixelCornerConv(
                    matches[i].pixel_coords + Vec2f::Constant(0.5f) + Vec2f(kFramingExtent, kFramingExtent),
                    matches[i].pixel_coords + Vec2f::Constant(0.5f) + Vec2f(-kFramingExtent, kFramingExtent),
                    Vec3u8(255, 255, 255));
                d->debug_display->AddSubpixelLinePixelCornerConv(
                    matches[i].pixel_coords + Vec2f::Constant(0.5f) + Vec2f(-kFramingExtent, kFramingExtent),
                    matches[i].pixel_coords + Vec2f::Constant(0.5f) + Vec2f(-kFramingExtent, -kFramingExtent),
                    Vec3u8(255, 255, 255));
                d->debug_display->AddSubpixelLinePixelCornerConv(
                    matches[i].pixel_coords + Vec2f::Constant(0.5f) + Vec2f(-kFramingExtent, -kFramingExtent),
                    matches[i].pixel_coords + Vec2f::Constant(0.5f) + Vec2f(kFramingExtent, -kFramingExtent),
                    Vec3u8(255, 255, 255));
                d->debug_display->AddSubpixelLinePixelCornerConv(
                    matches[i].pixel_coords + Vec2f::Constant(0.5f) + Vec2f(kFramingExtent, -kFramingExtent),
                    matches[i].pixel_coords + Vec2f::Constant(0.5f) + Vec2f(kFramingExtent, kFramingExtent),
                    Vec3u8(255, 255, 255));
              }
              
              if (debug_step_by_step) {
                d->debug_display->Update();
                std::getchar();
              }
            }
            
            if (debug) {
              LOG(WARNING) << "Debug: removing a bad point (homography prediction too far off in neighbor check)";
            }
            pattern_feature_detections.erase(Vec2i(new_detection.y(), new_detection.z()));
            pattern_feature_rejections.insert(Vec2i(new_detection.y(), new_detection.z()));
            break;  // proceed to next new detection to extend from
          }
          
          // Perform length and angle threshold based outlier filtering.
          auto opposite_it = pattern_feature_detections.find(Vec2i(new_detection.y() - x_directions[direction_index],
                                                                   new_detection.z() - y_directions[direction_index]));
          if (opposite_it != pattern_feature_detections.end()) {
            if (!CheckOppositeAngleAndLengthCriterion(
                neighbor_it->second.position,
                feature_detection.position,
                opposite_it->second.position,
                debug)) {
              if (debug) {
                LOG(WARNING) << "Debug: removing a bad point (length/angle criterion off in neighbor check)";
              }
              pattern_feature_detections.erase(neighbor_coords);
              pattern_feature_rejections.insert(neighbor_coords);
              pattern_feature_detections.erase(Vec2i(new_detection.y(), new_detection.z()));
              pattern_feature_rejections.insert(Vec2i(new_detection.y(), new_detection.z()));
              break;
            } else {
              // ++ feature_detection.num_validations;
            }
          }
          
          if (debug) {
            d->debug_display->AddSubpixelLinePixelCornerConv(
                neighbor_it->second.position + Vec2f::Constant(0.5f),
                feature_detection.position + Vec2f::Constant(0.5f),
                debug_colors[pattern.tags[0].index % 8]);
          }
        }
      }  // loop over neighbors to extend to
    }  // loop over new detections to extend from
  }  // while (num_predictions > 0)
}

void FeatureDetectorTaggedPattern::PrepareCUDAResources(int image_width, int image_height) {
  if (!d->cuda_image ||
      d->cuda_image->width() < image_width ||
      d->cuda_image->height() < image_height) {
    d->cuda_image.reset(new CUDABuffer<u8>(image_height, image_width));
    d->cuda_gradient_image.reset(new CUDABuffer<float2>(image_height, image_width));
  }
  
  // TODO: Need a way to handle refinements with more features. Those could be split up, the buffer could be re-allocated, or this could be made configurable.
  //       This applies to e.g., d->cost_buffer and d->position_temp_buffer.
  int max_feature_count_in_refinement = 128;
  
  int cost_buffer_size = max_feature_count_in_refinement * window_half_extent * window_half_extent;
  if (!d->cost_buffer ||
      d->cost_buffer_size < cost_buffer_size) {
    cudaFree(d->cost_buffer);
    cudaMalloc(&d->cost_buffer, cost_buffer_size * sizeof(cub::KeyValuePair<int, float>));
    d->cost_buffer_size = cost_buffer_size;
  }
  
  // Upload sample locations into global buffer
  // TODO: Try using a "constant buffer" for even more speed?
  if (!d->samples_gpu ||
      d->samples_gpu->width() != d->samples.size()) {
    d->samples_gpu.reset(new CUDABuffer<float2>(1, d->samples.size()));
    d->samples_gpu->UploadAsync(/*stream*/ 0, reinterpret_cast<float2*>(d->samples.data()));
    d->pattern_samples_gpu.reset(new CUDABuffer<float2>(max_feature_count_in_refinement, d->samples.size()));
    d->rendered_samples.reset(new CUDABuffer<float>(max_feature_count_in_refinement, d->samples.size()));
  }
  
  if (!d->position_temp_buffer) {
    d->position_temp_buffer.reset(new CUDABuffer<float2>(1, max_feature_count_in_refinement));
    d->state_buffer.reset(new CUDABuffer<float4>(1, max_feature_count_in_refinement));
    d->local_pattern_tr_pixel_buffer.reset(new CUDABuffer<float>(1, 9 * max_feature_count_in_refinement));
  }
  
  if (!d->cuda_objects_initialized) {
    d->cuda_image->CreateTextureObject(
        cudaAddressModeClamp,
        cudaAddressModeClamp,
        cudaFilterModeLinear,
        cudaReadModeNormalizedFloat,
        /*use_normalized_coordinates*/ false,
        &d->image_texture);
    
    d->cuda_gradient_image->CreateTextureObject(
        cudaAddressModeClamp,
        cudaAddressModeClamp,
        cudaFilterModeLinear,
        cudaReadModeElementType,
        /*use_normalized_coordinates*/ false,
        &d->gradient_image_texture);
  }
  
  d->cuda_objects_initialized = true;
}

void FeatureDetectorTaggedPattern::PrepareAprilTagDetector() {
  if (!d->apriltag_detector) {
    d->apriltag_detector = apriltag_detector_create();
    d->apriltag_family = tag36h11_create();
    apriltag_detector_add_family(d->apriltag_detector, d->apriltag_family);
  }
}

void FeatureDetectorTaggedPattern::RefineFeatureDetections(
    const Image<u8>& image,
    const Image<Vec2f>& gradient_image,
    const Image<float>& gradmag_image,
    int num_features,
    const FeatureDetection* predicted_features,
    FeatureDetection* output,
    bool debug,
    bool debug_step_by_step) {
  // For features that are rejected, the final_cost member will be set to a
  // negative value.
  vector<FeatureDetection> filtered_predictions;
  filtered_predictions.reserve(num_features);
  vector<int> filtered_to_original_index;
  filtered_to_original_index.reserve(num_features);
  
  // Filter out feature predictions which are too close to the image or pattern borders.
  // TODO: Do a second pass to check for being too close to the pattern borders after the refinement?
  for (int i = 0; i < num_features; ++ i) {
    output[i].final_cost = -1;
    
    // Too close to image border?
    auto& position = predicted_features[i].position;
    if (!(position.x() - window_half_extent >= 0 &&
          position.y() - window_half_extent >= 0 &&
          position.x() + window_half_extent < image.width() - 1 &&
          position.y() + window_half_extent < image.height() - 1)) {
      continue;
    }
    
    // Out of pattern?
    // TODO: Use the correct pattern in these checks instead of always using d->patterns[0].
    Mat3f local_pattern_tr_pixel = predicted_features[i].local_pixel_tr_pattern.inverse();
    
    bool in_pattern = true;
    for (int corner = 0; corner < 4; ++ corner) {
      Vec2f pixel_offset(((corner % 2 == 0) ? 1 : -1) * window_half_extent,
                         ((corner / 2 == 0) ? 1 : -1) * window_half_extent);
      Vec2f pattern_offset = Vec3f(local_pattern_tr_pixel * pixel_offset.homogeneous()).hnormalized();
      Vec2f pattern_sample_coordinate = predicted_features[i].pattern_coordinate.cast<float>() + pattern_offset;
      if (!d->patterns[0].IsValidPatternCoord(pattern_sample_coordinate.x(), pattern_sample_coordinate.y())) {
        in_pattern = false;
        break;
      }
    }
    if (!in_pattern) {
      continue;
    }
    
    filtered_predictions.push_back(predicted_features[i]);
    
    filtered_to_original_index.push_back(i);
  }
  
  vector<Vec2f> position_after_intensity_based_refinement(filtered_predictions.size());
  
  const int num_intensity_samples = (1 / 8.) * d->samples.size();
  const int num_gradient_samples = d->samples.size();
  if (d->use_cuda) {
    if (refinement_type != FeatureRefinement::GradientsXY && refinement_type != FeatureRefinement::Intensities) {
      LOG(FATAL) << "CUDA-based feature refinement only supports the gradients_xy and intensities refinement types.";
    }
    
    vector<FeatureDetection> filtered_detections(filtered_predictions.size());
    
    if (!filtered_predictions.empty()) {
      CHECK_CUDA_NO_ERROR();
      
      constexpr cudaStream_t stream = 0;
      
      // Refinement by matching with the known pattern
      RefineFeatureByMatchingCUDA(
          stream,
          d->patterns[0].num_star_segments,  // TODO: Use the correct pattern here instead of always the one with index 0
          num_intensity_samples, *d->samples_gpu, d->image_texture,
          window_half_extent, image.width(), image.height(),
          filtered_predictions,
          &position_after_intensity_based_refinement,
          d->cost_buffer_size, d->state_buffer.get(), d->rendered_samples.get(),
          d->local_pattern_tr_pixel_buffer.get());
      
      // Symmetry-based refinement using gradients
      // NOTE: It would not be necessary to transfer intensity_refined_positions back
      //       to the GPU, as the next call does, since these are the results of the
      //       previous call, which should already be on the GPU.
      RefineFeatureBySymmetryCUDA(
          stream, num_gradient_samples, *d->samples_gpu, d->pattern_samples_gpu.get(),
          (refinement_type == FeatureRefinement::GradientsXY) ? d->gradient_image_texture : d->image_texture,
          refinement_type, window_half_extent, image.width(), image.height(),
          filtered_predictions,
          position_after_intensity_based_refinement,
          &filtered_detections,
          d->cost_buffer_size, d->cost_buffer,
          d->local_pattern_tr_pixel_buffer.get());
    }
    
    for (usize i = 0; i < filtered_predictions.size(); ++ i) {
      FeatureDetection& this_output = output[filtered_to_original_index[i]];
      
      if (std::isnan(filtered_detections[i].position.x()) || std::isinf(filtered_detections[i].position.x())) {
        this_output.final_cost = -1;
      } else {
        this_output = filtered_detections[i];
      }
    }
  } else {
    for (usize i = 0; i < filtered_predictions.size(); ++ i) {
      FeatureDetection& this_output = output[filtered_to_original_index[i]];
      
      const Mat3f& local_pixel_tr_pattern = filtered_predictions[i].local_pixel_tr_pattern;
      Mat3f local_pattern_tr_pixel = local_pixel_tr_pattern.inverse();
      
      if (!RefineFeatureByMatching(
          num_intensity_samples,
          d->samples,
          image,
          window_half_extent,
          filtered_predictions[i].position,
          local_pattern_tr_pixel,
          d->patterns[0],  // TODO: Use the correct pattern here instead of always the one with index 0
          &this_output.position,
          nullptr,
          debug)) {
        // Could not find a corner here.
        if (debug) {
          d->debug_display->AddSubpixelDotPixelCornerConv(filtered_predictions[i].position + Vec2f::Constant(0.5f), Vec3u8(255, 0, 0));
          if (debug_step_by_step) {
            LOG(WARNING) << "Failure during matching-based refinement";
            d->debug_display->Update();
            std::getchar();
          }
        }
        this_output.final_cost = -1;
        continue;
      }
      
      position_after_intensity_based_refinement[i] = this_output.position;
      
      bool feature_found = false;
      if (refinement_type == FeatureRefinement::GradientsXY) {
        feature_found = RefineFeatureBySymmetry<SymmetryCostFunction_GradientsXY>(
            num_gradient_samples,
            d->samples,
            gradient_image,
            window_half_extent,
            this_output.position,
            local_pattern_tr_pixel,
            local_pixel_tr_pattern,
            &this_output.position,
            &this_output.final_cost,
            debug);
      } else if (refinement_type == FeatureRefinement::GradientMagnitude) {
        feature_found = RefineFeatureBySymmetry<SymmetryCostFunction_SingleChannel>(
            num_gradient_samples,
            d->samples,
            gradmag_image,
            window_half_extent,
            this_output.position,
            local_pattern_tr_pixel,
            local_pixel_tr_pattern,
            &this_output.position,
            &this_output.final_cost,
            debug);
      } else if (refinement_type == FeatureRefinement::Intensities) {
        feature_found = RefineFeatureBySymmetry<SymmetryCostFunction_SingleChannel>(
            num_gradient_samples,
            d->samples,
            image,
            window_half_extent,
            this_output.position,
            local_pattern_tr_pixel,
            local_pixel_tr_pattern,
            &this_output.position,
            &this_output.final_cost,
            debug);
      } else if (refinement_type == FeatureRefinement::NoRefinement) {
        // Use the output of the matching-based feature detection as-is.
        this_output.final_cost = 0;
        feature_found = true;
      } else {
        LOG(FATAL) << "Unsupported feature refinement type";
      }
      
      if (!feature_found) {
        // Could not find a feature here.
        if (debug) {
          d->debug_display->AddSubpixelDotPixelCornerConv(filtered_predictions[i].position + Vec2f::Constant(0.5f), Vec3u8(255, 0, 0));
          if (debug_step_by_step) {
            LOG(WARNING) << "Failure during symmetry-based refinement";
            d->debug_display->Update();
            std::getchar();
          }
        }
        this_output.final_cost = -1;
        continue;
      }
    }
  }
  
  // Filter outliers based on the difference between the two refined positions.
  for (usize i = 0; i < filtered_predictions.size(); ++ i) {
    FeatureDetection& this_output = output[filtered_to_original_index[i]];
    
    if (this_output.final_cost >= 0 &&
        (this_output.position - position_after_intensity_based_refinement[i]).squaredNorm() > 0.75f) {  // TODO: make parameter
      // If the position after raw intensities based refinement is too different
      // from the position after gradient based refinement, the recording conditions
      // are probably bad.
      if (debug) {
        d->debug_display->AddSubpixelDotPixelCornerConv(position_after_intensity_based_refinement[i] + Vec2f::Constant(0.5f), Vec3u8(200, 20, 20));
        d->debug_display->AddSubpixelDotPixelCornerConv(this_output.position + Vec2f::Constant(0.5f), Vec3u8(20, 150, 20));
        if (debug_step_by_step) {
          LOG(WARNING) << "Corner rejected because of large difference between raw-intensity and gradient refinement: "
                       << (this_output.position - position_after_intensity_based_refinement[i]).norm() << " px";
          d->debug_display->Update();
          std::getchar();
        }
      }
      this_output.final_cost = -1;
    }
  }
}

}
