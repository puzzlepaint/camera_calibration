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

#pragma once

#include <forward_list>
#include <iomanip>
#include <unordered_set>

#include <libvis/libvis.h>
#include <libvis/image.h>

#include "camera_calibration/feature_detection/feature_detector.h"

struct zarray;

namespace vis {

/// Information about the placement of an AprilTag within a known pattern.
struct AprilTagInfo {
  /// (x, y) gives the square coordinate of the top-left corner of the AprilTag
  /// (the corresponding feature coordinate would be (x - 1, y - 1), however
  /// that feature is not used since its context region is partly covered by
  /// the tag).
  int x;
  
  /// See comment on x.
  int y;
  
  /// Width of the AprilTag in squares.
  int width;
  
  /// Height of the AprilTag in squares.
  int height;
  
  /// Index of the AprilTag in its tag family, used to uniquely identify the
  /// tag within its family.
  int index;
};


/// Characteristics of a known pattern that can be detected by the detector.
struct PatternData {
  /// Returns whether the given integer coordinate is a valid feature coordinate
  /// of the pattern.
  inline bool IsValidFeatureCoord(int x, int y) const {
    if (!(x >= 0 &&
          y >= 0 &&
          x <= squares_x - 2 &&
          y <= squares_y - 2)) {
      return false;
    }
    for (const AprilTagInfo& tag : tags) {
      if (x >= tag.x - 1 &&
          y >= tag.y - 1 &&
          x <= tag.x - 1 + tag.width &&
          y <= tag.y - 1 + tag.height) {
        return false;
      }
    }
    return true;
  }
  
  /// Returns whether the floating-point coordinate in the feature coordinate
  /// system is within the area of the calibration target that is covered by
  /// the repeating pattern.
  inline bool IsValidPatternCoord(float x, float y) const {
    if (!(x >= -1.f &&
          y >= -1.f &&
          x <= squares_x - 1.f &&
          y <= squares_y - 1.f)) {
      return false;
    }
    for (const AprilTagInfo& tag : tags) {
      if (x >= tag.x - 1 &&
          y >= tag.y - 1 &&
          x <= tag.x - 1 + tag.width &&
          y <= tag.y - 1 + tag.height) {
        return false;
      }
    }
    return true;
  }
  
  /// Returns the pattern intensity (0 for black, 1 for white, 0.5 for ill-defined
  /// positions) at the given position within the pattern. The pattern is supposed
  /// to have endless extent, feature positions are at integer coordinates, and
  /// (0, 0) is supposed to correspond to a feature location.
  template <typename Derived>
  inline float PatternIntensityAt(const MatrixBase<Derived>& position) const {
    // Have coordinates in [-0.5, 0.5].
    Vec2f c;
    c.x() = -1 * (position.x() - (position.x() > 0 ? 1 : -1) * static_cast<int>(std::fabs(position.x()) + 0.5f));
    c.y() = position.y() - (position.y() > 0 ? 1 : -1) * static_cast<int>(std::fabs(position.y()) + 0.5f);
    
    if (c.squaredNorm() < 1e-8f) {
      return 0.5f;
    }
    
    float angle = std::atan2(c.y(), c.x()) - 0.5f * M_PI;
    if (angle < 0) {
      angle += 2 * M_PI;
    }
    return (static_cast<int>(num_star_segments * angle / (2 * M_PI)) % 2 == 0) ? 1.f : 0.f;
  }
  
  /// Computes the number of features within this pattern.
  int ComputeFeatureCount() const {
    int feature_count = (squares_x - 1) * (squares_y - 1);
    
    // Since the tags might be partly outside the area where features are, we
    // need to pay attention to how large the overlap is.
    // NOTE: We do not consider the theoretical possibility that AprilTags
    //       might be directly next to each other, thus blocking one row or
    //       column less than they would otherwise.
    for (const AprilTagInfo& tag : tags) {
      int tag_min_feature_x = std::max(tag.x - 1, 0);
      int tag_min_feature_y = std::max(tag.y - 1, 0);
      int tag_max_feature_x = std::min(tag.x - 1 + tag.width, squares_x - 1);
      int tag_max_feature_y = std::min(tag.y - 1 + tag.height, squares_y - 1);
      
      feature_count -= (tag_max_feature_x - tag_min_feature_x + 1) *
                       (tag_max_feature_y - tag_min_feature_y + 1);
    }
    
    return feature_count;
  }
  
  /// Returns a corner position of the given star triangle.
  Vec2f GetStarCoord(float square_length, float i, float center_x, float center_y) const {
    float angle = ((2 * M_PI) * i) / num_star_segments;
    float x = sin(angle);
    float y = cos(angle);
    
    float max_abs_x = max(abs(x), abs(y));
    x /= max_abs_x;
    y /= max_abs_x;
    
    return Vec2f(center_x - 0.5 * square_length * x,
                 center_y - 0.5 * square_length * y);
  }
  
  /// Computes the geometry of all black areas of the pattern as polygons.
  /// Feature positions are at integer coordinates, and (0, 0) is
  /// supposed to correspond to a feature location.
  /// The AprilTags are not included in the returned geometry, unless the path
  /// to the AprilTag files is given as an optional parameter.
  void ComputePatternGeometry(forward_list<vector<Vec2f>>* geometry, const char* apriltags_path = nullptr) const {
    constexpr float square_length = 1;
    
    geometry->clear();
    
    for (int y = -1; y < squares_y; ++ y) {
      for (int x = -1; x < squares_x; ++ x) {
        bool is_in_apriltag = false;
        for (const AprilTagInfo& tag : tags) {
          if (x >= tag.x &&
              y >= tag.y &&
              x <= tag.x - 2 + tag.width &&
              y <= tag.y - 2 + tag.height) {
            is_in_apriltag = true;
            break;
          }
        }
        if (is_in_apriltag) {
          continue;
        }
        
        // Compute the black stripes around this feature.
        for (int segment = 0; segment < num_star_segments; segment += 2) {
          Vec2f middle_coord = GetStarCoord(square_length, segment + 0.5f, x, y);
          if (!IsValidPatternCoord(middle_coord.x(), middle_coord.y())) {
            continue;
          }
          
          geometry->emplace_front();
          vector<Vec2f>* polygon = &geometry->front();
          
          polygon->emplace_back(x, y);
          polygon->emplace_back(GetStarCoord(square_length, segment, x, y));
          
          // Add point at the square corner?
          float angle1 = (2 * M_PI) * (segment) / num_star_segments;
          float angle2 = (2 * M_PI) * (segment + 1) / num_star_segments;
          if (floor((angle1 - M_PI / 4) / (M_PI / 2)) != floor((angle2 - M_PI / 4) / (M_PI / 2))) {
            float corner_angle = (M_PI / 4) + (M_PI / 2) * floor((angle2 - M_PI / 4) / (M_PI / 2));
            float corner_x = sin(corner_angle);
            float corner_y = cos(corner_angle);
            float normalizer = abs(corner_x);
            corner_x /= normalizer;
            corner_y /= normalizer;
            polygon->emplace_back(Vec2f(
                x - 0.5 * square_length * corner_x,
                y - 0.5 * square_length * corner_y));
          }
          
          polygon->emplace_back(GetStarCoord(square_length, segment + 1, x, y));
        }
      }
    }
    
    if (apriltags_path) {
      for (const AprilTagInfo& tag : tags) {
        ostringstream tag_filename;
        tag_filename << "tag36_11_" << std::setw(5) << std::setfill('0') << tag.index << ".png";
        string tag_path = string(apriltags_path) + tag_filename.str();
        Image<u8> tag_image(tag_path);
        if (tag_image.empty()) {
          LOG(ERROR) << "Cannot read image: " << tag_path;
          continue;
        }
        
        for (int y = 0; y < tag_image.height(); ++ y) {
          for (int x = 0; x < tag_image.width(); ++ x) {
            if (tag_image(x, y) < 127) {
              geometry->emplace_front();
              vector<Vec2f>* polygon = &geometry->front();
              
              // Change coordinate system for tag x/y
              int tag_x = tag.x - 1;
              int tag_y = tag.y - 1;
              
              polygon->emplace_back(tag_x + x * tag.width / (1. * tag_image.width()),
                                    tag_y + y * tag.height / (1. * tag_image.height()));
              polygon->emplace_back(tag_x + (x + 1) * tag.width / (1. * tag_image.width()),
                                    tag_y + y * tag.height / (1. * tag_image.height()));
              polygon->emplace_back(tag_x + (x + 1) * tag.width / (1. * tag_image.width()),
                                    tag_y + (y + 1) * tag.height / (1. * tag_image.height()));
              polygon->emplace_back(tag_x + x * tag.width / (1. * tag_image.width()),
                                    tag_y + (y + 1) * tag.height / (1. * tag_image.height()));
            }
          }
        }
      }
    }
  }
  
  /// Number of squares in x direction.
  /// TODO: The term "square" is probably not applicable anymore to the star pattern?
  int squares_x;
  
  /// Number of squares in y direction.
  int squares_y;
  
  /// Number of segments (both black and white segments are counted) in each
  /// "star" around each feature. For example, for a checkerboard this would
  /// be 4.
  int num_star_segments;
  
  /// Information about the AprilTags contained in this pattern that are used
  /// to localize it.
  vector<AprilTagInfo> tags;
  
  /// Page properties. These are not relevant for feature detection, but are
  /// helpful if one wants to render the pattern synthetically.
  float page_width_mm;
  float page_height_mm;
  float pattern_start_x_mm;
  float pattern_start_y_mm;
  float pattern_end_x_mm;
  float pattern_end_y_mm;
};


/// Data of a single detected feature.
struct FeatureDetection {
  /// Subpixel position in the image ("pixel center" coordinate convention)
  Vec2f position;
  
  // Integer coordinate of the feature in the pattern
  Vec2i pattern_coordinate;
  
  // The direction that was used from an older existing feature detection to
  // arrive at this feature detection. May be (0, 0) if the feature was
  // predicted based on an AprilTag. May be uninitialized for final feature
  // detections; this is used for feature predictions only.
  Vec2i extension_direction;
  
  /// Final cost value of the feature refinement optimization process. This
  /// may be used for outlier detection.
  float final_cost;
  
  /// Number of times this feature has been validated.
  // int num_validations = 0;  // NOTE: Currently unused
  
  /// Homography on local coordinate systems. Is supposed to map the integer
  /// feature coordinate in the pattern (where (0, 0) is defined to be the
  /// feature corresponding to this detection in the local coordinate system) to
  /// the pixel position (where pixel (0, 0) is defined to be at the predicted
  /// feature position).
  Mat3f local_pixel_tr_pattern;
};


struct FeatureDetectorTaggedPatternPrivate;

/// Feature detector for "star" patterns with AprilTags on them.
class FeatureDetectorTaggedPattern : public FeatureDetector {
 public:
  /// Constructs a new detector. Check valid() afterwards to test whether the
  /// given YAML file(s) were loaded correctly.
  FeatureDetectorTaggedPattern(
      const vector<string>& pattern_yaml_paths,
      int window_half_extent,
      FeatureRefinement refinement_type,
      bool use_cuda);
  
  ~FeatureDetectorTaggedPattern();
  
  /// See documentation in base class.
  virtual bool SetPatternYAMLPaths(
      const vector<string>& paths) override;
  
  /// See documentation in base class.
  virtual void DetectFeatures(
      const Image<Vec3u8>& image,
      vector<PointFeature>* features,
      Image<Vec3u8>* detection_visualization) override;
  
  /// See documentation in base class.
  virtual int GetPatternCount() const override;
  
  /// Returns the PatternData for the given pattern index.
  /// The valid range for pattern_index is [0, GetPatternCount() - 1].
  const PatternData& GetPatternData(int pattern_index) const;
  
  /// See documentation in base class.
  virtual void GetCorners(
      int pattern_index,
      unordered_map<int, Vec2i>* feature_id_to_coord) const override;
  
  inline bool valid() const { return valid_; }
  
 private:
  void PredictFeaturesNextToAprilTags(
      const Image<u8>& image,
      zarray* detections,
      float tag_upscale_factor,
      vector<unordered_map<Vec2i, FeatureDetection>>* feature_predictions,
      bool debug,
      Vec3u8 debug_colors[8]);
  
  void PredictAndDetectFeatures(
      const Image<u8>& image,
      const Image<Vec2f>& gradient_image,
      const Image<float>& gradmag_image,
      vector<unordered_map<Vec2i, FeatureDetection>>* feature_predictions,
      vector<unordered_map<Vec2i, FeatureDetection>>* feature_detections,
      vector<unordered_set<Vec2i>>* feature_rejections,
      bool debug,
      bool debug_step_by_step,
      Vec3u8 debug_colors[8]);
  
  void PrepareCUDAResources(
      int image_width,
      int image_height);
  
  void PrepareAprilTagDetector();
  
  void RefineFeatureDetections(
      const Image<u8>& image,
      const Image<Vec2f>& gradient_image,
      const Image<float>& gradmag_image,
      int num_features,
      const FeatureDetection* predicted_features,
      FeatureDetection* output,
      bool debug,
      bool debug_step_by_step);
  
  unique_ptr<FeatureDetectorTaggedPatternPrivate> d;
  bool valid_;
};

}
