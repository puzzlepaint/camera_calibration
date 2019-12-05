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

#include "camera_calibration/calibration_initialization/dense_initialization.h"

#include <libvis/dlt.h>
#include <libvis/image_display.h>
#include <libvis/point_cloud.h>
#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac/SampleConsensusProblem.hpp>
#include <opengv/sac/SampleConsensus.hpp>
#include <opengv/sac_problems/point_cloud/PointCloudSacProblem.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>

#include "camera_calibration/relative_pose_initialization/algorithms.h"
#include "camera_calibration/dataset.h"
#include "camera_calibration/hash_vec2i.h"
#include "camera_calibration/ui/calibration_window.h"
#include "camera_calibration/util.h"


namespace vis {

/// Groups commonly passed parameters for convenience
struct DenseInitializationParameters {
  Dataset* dataset;
  int camera_index;
  CalibrationWindow* calibration_window;
  bool step_by_step;
};


struct Square {
  // The point must be given in pixel-corner origin convention.
  inline bool IsImagePointInSquare(const Vec2f& point) const {
    bool inside = false;
    for (int i = 0; i < 4; ++ i) {
      if (point.y() >= min_y[i] &&
          point.y() < max_y[i] &&
          point.dot(normal_vectors[i]) < d[i]) {
        inside = !inside;
      }
    }
    return inside;
  }
  
  Vec2f InterpolatePatternCoordinatesAt(const Vec2f& point) const {
    return (Vec3f(image_to_pattern_homography * point.homogeneous()).hnormalized());
  }
  
  void Initialize() {
    total_max_y = -numeric_limits<double>::infinity();
    total_min_y = numeric_limits<double>::infinity();
    
    for (int i = 0; i < 4; ++ i) {
      int next_i = (i+1)%4;
      
      min_y[i] = std::min(image_coordinates[i].y(), image_coordinates[next_i].y());
      total_min_y = std::min(total_min_y, min_y[i]);
      max_y[i] = std::max(image_coordinates[i].y(), image_coordinates[next_i].y());
      total_max_y = std::max(total_max_y, max_y[i]);
      normal_vectors[i] = Vec2f(
          - (image_coordinates[next_i].y() - image_coordinates[i].y()),
          image_coordinates[next_i].x() - image_coordinates[i].x());
      // Make all normals point to the right.
      if (normal_vectors[i].x() < 0) {
        normal_vectors[i] = -normal_vectors[i];
      }
      d[i] = image_coordinates[i].dot(normal_vectors[i]);
    }
    
    image_to_pattern_homography = NormalizedDLT(image_coordinates, pattern_coordinates, 4);
  }
  
  /// Using pixel-corner origin convention.
  Vec2f image_coordinates[4];
  /// Coordinate of the point in pattern coordinates.
  Vec2f pattern_coordinates[4];
  
  Mat3f image_to_pattern_homography;
  
  Vec2f normal_vectors[4];
  double d[4];
  double max_y[4];
  double min_y[4];
  double total_max_y;
  double total_min_y;
};

/// Returns the number of dense matches.
int DensifyMatches(
    const vector<PointFeature>& matches,
    Dataset* dataset,
    int camera_index,
    const vector<bool>& known_geometry_localized,
    const vector<Mat3f>& global_r_known_geometry,
    const vector<Vec3f>& global_t_known_geometry,
    int buffer_width,
    int buffer_height,
    Image<Vec3d>* dense_matches) {
  int match_count = 0;
  int square_overlaps = 0;
  int num_near_parallel = 0;
  
  dense_matches->SetSize(buffer_width, buffer_height);
  dense_matches->SetTo(Vec3d::Constant(numeric_limits<double>::quiet_NaN()));
  
  for (int known_geometry_index = 0; known_geometry_index < static_cast<int>(known_geometry_localized.size()); ++ known_geometry_index) {
    if (!known_geometry_localized[known_geometry_index]) {
      continue;
    }
    
    vector<Square> squares;
    
    const KnownGeometry& kg = dataset->GetKnownGeometry(known_geometry_index);
    const Mat3f& global_r_kg = global_r_known_geometry[known_geometry_index];
    const Vec3f& global_t_kg = global_t_known_geometry[known_geometry_index];
    
    // Index matches by integer coordinate in the calibration target
    unordered_map<Vec2i, usize> coord_to_match_index;  // indexed by kg.feature_id_to_position[match.id], returns index in vector<PointFeature>& matches.
    for (usize m = 0; m < matches.size(); ++ m) {
      const PointFeature& match = matches[m];
      
      auto it = kg.feature_id_to_position.find(match.id);
      if (it != kg.feature_id_to_position.end()) {
        coord_to_match_index.insert(make_pair(it->second, m));
      }
    }
    
    // Find squares among the detected features of the chosen known geometry.
    for (const pair<Vec2i, usize>& item : coord_to_match_index) {
      auto it_right = coord_to_match_index.find(item.first + Vec2i(1, 0));
      if (it_right == coord_to_match_index.end()) {
        continue;
      }
      auto it_bottom_right = coord_to_match_index.find(item.first + Vec2i(1, 1));
      if (it_bottom_right == coord_to_match_index.end()) {
        continue;
      }
      auto it_bottom = coord_to_match_index.find(item.first + Vec2i(0, 1));
      if (it_bottom == coord_to_match_index.end()) {
        continue;
      }
      
      squares.emplace_back();
      Square& new_square = squares.back();
      new_square.image_coordinates[0] = matches[item.second].xy;
      new_square.pattern_coordinates[0] = item.first.cast<float>() * kg.cell_length_in_meters;
      new_square.image_coordinates[1] = matches[it_right->second].xy;
      new_square.pattern_coordinates[1] = it_right->first.cast<float>() * kg.cell_length_in_meters;
      new_square.image_coordinates[2] = matches[it_bottom_right->second].xy;
      new_square.pattern_coordinates[2] = it_bottom_right->first.cast<float>() * kg.cell_length_in_meters;
      new_square.image_coordinates[3] = matches[it_bottom->second].xy;
      new_square.pattern_coordinates[3] = it_bottom->first.cast<float>() * kg.cell_length_in_meters;
      new_square.Initialize();
    }
    
    // Iterate over all squares. For each square, iterate over all pixels
    // covered by it. For those, use a homography to obtain an estimate for the
    // calibration target position which matches the pixel.
    double scale_x = dataset->GetImageSize(camera_index).x() / static_cast<double>(buffer_width);
    double scale_y = dataset->GetImageSize(camera_index).y() / static_cast<double>(buffer_height);
    for (const Square& square : squares) {
      int min_x = numeric_limits<int>::max();
      int min_y = numeric_limits<int>::max();
      int max_x = 0;
      int max_y = 0;
      
      for (int i = 0; i < 4; ++ i) {
        int x = square.image_coordinates[i].x() / scale_x;
        int y = square.image_coordinates[i].y() / scale_y;
        min_x = std::min(min_x, x);
        min_y = std::min(min_y, y);
        max_x = std::max(max_x, x);
        max_y = std::max(max_y, y);
      }
      
      for (int y = min_y; y <= max_y; ++ y) {
        double scaled_y = scale_y * (y + 0.5f);
        if (scaled_y < square.total_min_y ||
            scaled_y >= square.total_max_y) {
          continue;
        }
        
        double min_filled_x = -1;
        double max_filled_x = -1;
        bool near_parallel = false;
        for (int i = 0; i < 4; ++ i) {
          if (scaled_y < square.min_y[i] ||
              scaled_y >= square.max_y[i]) {
            continue;
          }
          
          int next_i = (i+1)%4;
          
          // Equation to get the line parameter for the intersection of this square
          // edge with the horizontal line y == scaled_y:
          // square.image_coordinates[i].y() + intersection_line_parameter * (square.image_coordinates[next_i].y() - square.image_coordinates[i].y()) == scaled_y;
          double denominator = square.image_coordinates[next_i].y() - square.image_coordinates[i].y();
          if (fabs(denominator) < 1e-6) {
            near_parallel = true;
            break;
          }
          double intersection_line_parameter = (scaled_y - square.image_coordinates[i].y()) / denominator;
          double intersection_x = square.image_coordinates[i].x() + intersection_line_parameter * (square.image_coordinates[next_i].x() - square.image_coordinates[i].x());
          
          if (min_filled_x < 0) {
            min_filled_x = intersection_x;
          } else if (max_filled_x < 0) {
            max_filled_x = intersection_x;
          } else {
            near_parallel = true;
            break;
          }
        }
        
        if (!near_parallel) {
          if (min_filled_x > max_filled_x) {
            std::swap(min_filled_x, max_filled_x);
          }
          int min_filled_x_int = static_cast<int>(min_filled_x / scale_x + 0.5);
          int max_filled_x_int = static_cast<int>(max_filled_x / scale_x - 0.5);
          
          for (int x = min_filled_x_int; x <= max_filled_x_int; ++ x) {
            Vec2f point = Vec2f(scale_x * (x + 0.5f), scaled_y);
            if (std::isnan(dense_matches->at(x, y).x())) {
              ++ match_count;
            } else {
              ++ square_overlaps;
            }
            Vec2f pattern_point = square.InterpolatePatternCoordinatesAt(point);
            dense_matches->at(x, y) = (global_r_kg * Vec3f(pattern_point.x(), pattern_point.y(), 0) + global_t_kg).cast<double>();
          }
        } else {
          ++ num_near_parallel;
          
          // Test each pixel individually.
          for (int x = min_x; x <= max_x; ++ x) {
            Vec2f point = Vec2f(scale_x * (x + 0.5f), scaled_y);
            if (square.IsImagePointInSquare(point)) {
              if (std::isnan(dense_matches->at(x, y).x())) {
                ++ match_count;
              } else {
                ++ square_overlaps;
              }
              
              Vec2f pattern_point = square.InterpolatePatternCoordinatesAt(point);
              dense_matches->at(x, y) = (global_r_kg * Vec3f(pattern_point.x(), pattern_point.y(), 0) + global_t_kg).cast<double>();
            }
          }
        }
      }
    }
  }
  
  if (square_overlaps > 100 * known_geometry_localized.size()) {  // only warn if there is a significant number of pixels with overlap
    LOG(WARNING) << "Square overlaps in match densification (should be zero): " << square_overlaps;
  }
  if (num_near_parallel > 100 * known_geometry_localized.size()) {
    LOG(WARNING) << "Near-parallel edges in match densification (should likely be low): " << square_overlaps;
  }
  return match_count;
}


bool LocalizePattern(
    const Image<Vec3d>& calibration_sum,
    const Image<u32>& calibration_count,
    const vector<Vec3d>& pattern,
    const vector<Vec2d>& features,
    int first_feature,
    int last_feature,
    int min_calibrated_match_count,
    SE3d* out_pattern_tr_image) {
  // If all feature detections are on a line (or there are less than 3 features),
  // do not attempt to estimate the pose.
  bool all_features_on_line = true;
  Vec2d first_point;
  Vec2d direction;
  int counter = 0;
  for (usize i = first_feature; i <= last_feature; ++ i) {
    if (!std::isnan(features[i].x())) {
      if (counter == 0) {
        first_point = features[i];
      } else if (counter == 1) {
        direction = (features[i] - first_point).normalized();
      } else {
        Vec2d other_direction = (features[i] - first_point).normalized();
        if (fabs(direction.dot(other_direction)) < 0.98f) {
          all_features_on_line = false;
          break;
        }
      }
      
      ++ counter;
    }
  }
  if (all_features_on_line) {
    return false;
  }
  
  int match_count = 0;
  for (usize i = first_feature; i <= last_feature; ++ i) {
    int x = features[i].x();
    int y = features[i].y();
    if (x >= 0 && y >= 0 && x < calibration_sum.width() && y < calibration_sum.height()) {
      if (calibration_count(x, y) > 0) {
        ++ match_count;
      }
    }
  }
  
  if (match_count < min_calibrated_match_count) {
    return false;
  }
  
  constexpr int kDownsampleCellSize = 15;
  Image<u8> downsample_occupancy_grid(
      calibration_sum.width() / kDownsampleCellSize,
      calibration_sum.height() / kDownsampleCellSize);
  downsample_occupancy_grid.SetTo(static_cast<u8>(0));
  
  opengv::bearingVectors_t bearing_vectors;
  bearing_vectors.reserve(downsample_occupancy_grid.pixel_count());
  opengv::points_t pattern_points;
  pattern_points.reserve(downsample_occupancy_grid.pixel_count());
  
  for (usize i = first_feature; i <= last_feature; ++ i) {
    int x = features[i].x();
    int y = features[i].y();
    if (x >= 0 && y >= 0 && x < calibration_sum.width() && y < calibration_sum.height()) {
      int gx = std::min<int>(downsample_occupancy_grid.width() - 1, x / kDownsampleCellSize);
      int gy = std::min<int>(downsample_occupancy_grid.height() - 1, y / kDownsampleCellSize);
      if (downsample_occupancy_grid(gx, gy) == 1) {
        continue;
      }
      downsample_occupancy_grid(gx, gy) = 1;
      
      u32 count = calibration_count(x, y);
      if (count > 0) {
        Vec3d bearing = calibration_sum(x, y) / count;
        bearing_vectors.push_back(bearing.normalized());
        pattern_points.push_back(pattern[i - first_feature]);
      }
    }
  }
  
  CHECK_EQ(bearing_vectors.size(), pattern_points.size());
  if (bearing_vectors.size() < 3) {
    return false;
  }
  opengv::absolute_pose::CentralAbsoluteAdapter adapter(
      bearing_vectors,
      pattern_points);
  
  std::shared_ptr<opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem> absposeproblem_ptr(
      new opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem(
          adapter, opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem::KNEIP));
  
  opengv::sac::Ransac<opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem> ransac;
  ransac.sac_model_ = absposeproblem_ptr;
  ransac.threshold_ = 1.0 - cos(atan(3 / 720.0));  // Equals a reprojection error of 3 pixels for focal length 720. TODO: Make this a parameter.
  ransac.max_iterations_ = 10;
  
  if (!ransac.computeModel()) {
    return false;
  }
  
  // Non-linear optimization (using all correspondences)
  adapter.sett(ransac.model_coefficients_.block<3, 1>(0, 3));
  adapter.setR(ransac.model_coefficients_.block<3, 3>(0, 0));
  opengv::transformation_t pattern_tr_image_matrix = opengv::absolute_pose::optimize_nonlinear(adapter);
  *out_pattern_tr_image = SE3d(
      Sophus::SE3d(pattern_tr_image_matrix.block<3, 3>(0, 0).cast<double>(),
                   pattern_tr_image_matrix.block<3, 1>(0, 3).cast<double>()));
  
  return true;
}


void DenseInitialization::LocalizeAdditionalPatterns(
    const DenseInitializationParameters& p,
    int imageset_index,
    const SE3d& global_tr_image,
    const Image<Vec3d>& calibration_sum,
    const Image<u32>& calibration_count) {
  vector<Vec3d> pattern;
  vector<Vec2d> features;
  
  double scale_x = static_cast<double>(calibration_sum.width()) / p.dataset->GetImageSize(p.camera_index).x();
  double scale_y = static_cast<double>(calibration_sum.height()) / p.dataset->GetImageSize(p.camera_index).y();
  for (usize k = 0; k < known_geometry_localized.size(); ++ k) {
    if (known_geometry_localized.at(k)) {
      continue;
    }
    const KnownGeometry& kg = p.dataset->GetKnownGeometry(k);
    
    pattern.clear();
    features.clear();
    const vector<PointFeature>& matches = p.dataset->GetImageset(imageset_index)->FeaturesOfCamera(p.camera_index);
    for (usize m = 0; m < matches.size(); ++ m) {
      const PointFeature& match = matches[m];
      
      auto it = kg.feature_id_to_position.find(match.id);
      if (it != kg.feature_id_to_position.end()) {
        Vec3d pattern_point = Vec3d(kg.cell_length_in_meters * it->second.x(), kg.cell_length_in_meters * it->second.y(), 0);
        pattern.push_back(pattern_point);
        features.push_back(Vec2f(scale_x, scale_y).cwiseProduct(match.xy).cast<double>());
      }
    }
    
    if (features.size() < 7) {  // TODO: arbitrary threshold, make configurable?
      continue;
    }
    
    // Try to localize the pattern.
    SE3d pattern_tr_image;
    if (LocalizePattern(
        calibration_sum,
        calibration_count,
        pattern,
        features,
        /*first_feature*/ 0,
        /*last_feature*/ features.size() - 1,
        /*min_calibrated_match_count*/ 7,
        &pattern_tr_image)) {
      SE3d global_tr_known_geometry = global_tr_image * pattern_tr_image.inverse();
      
      known_geometry_localized[k] = true;
      global_r_known_geometry[k] = global_tr_known_geometry.rotationMatrix().cast<float>();
      global_t_known_geometry[k] = global_tr_known_geometry.translation().cast<float>();
      
      LOG(INFO) << "Localized pattern #" << k;
      
      // TODO: Should update the estimated calibration with this new localized pattern instance
    }
  }
}


void DenseInitialization::AlternatingBundleAdjustment(
    const DenseInitializationParameters& p,
    Image<Vec3d>& calibration_sum,
    Image<u32>& calibration_count,
    int max_iterations) {
  vector<bool>& camera_image_used = image_used[p.camera_index];
  vector<SE3d>& camera_image_tr_global = image_tr_global[p.camera_index];
  
  for (int iteration = 0; iteration < max_iterations; ++ iteration) {
    // Optimize patterns, fixing intrinsics and image poses.
    // TODO: This requires to take all cameras initialized so far into account
    //       to prevent their image poses from becoming invalid.
    
    // Optimize image poses, fixing the intrinsics and patterns.
    // Afterwards, optimize intrinsics, fixing the image poses and patterns.
    Image<Vec3d> old_calibration_sum = calibration_sum;
    Image<u32> old_calibration_count = calibration_count;
    
    calibration_sum.SetTo(Vec3d::Zero());
    calibration_count.SetTo(static_cast<u32>(0));
    
    for (int imageset_index = 0; imageset_index < p.dataset->ImagesetCount(); ++ imageset_index) {
      if (!camera_image_used[imageset_index]) {
        continue;
      }
      
      // Attempt to localize the image using the old intrinsics.
      // NOTE: AttemptToLocalizeImage() always fills dense_matches, even if it
      //       returns false.
      SE3d global_tr_image;
      Image<Vec3d> dense_matches;
      if (AttemptToLocalizeImage(p, imageset_index, old_calibration_sum, old_calibration_count, &global_tr_image, &dense_matches)) {
        camera_image_tr_global[imageset_index] = global_tr_image.inverse();
      } else {
        LOG(WARNING) << "In alternating bundle adjustment: Unable to localize an image that had already been localized before.";
      }
      
      // Use the updated image pose to determine the new intrinsics
      UpdateCalibrationWithImage(
          camera_image_tr_global[imageset_index], dense_matches, &calibration_sum, &calibration_count);
    }
    
    if (p.calibration_window) {
      VisualizeIntrinsics(p, calibration_sum, &calibration_count, camera_image_used);
    }
  }
}


/// Cosmetic improvement: rotate the camera poses such that, in local camera coordinates,
/// (0, 0, 1) is forward, (1, 0, 0) is right, and (0, 1, 0) is down (given that a pinhole-like camera is used).
void ChooseNiceCameraOrientation(
    Image<Vec3d>& calibration_sum,
    Image<u32>& calibration_count,
    const vector<bool>& image_used,
    vector<SE3d>& image_tr_global) {
  // The min/max settings here define the image area that is averaged to obtain the corresponding direction.
  int forward_min_x = std::max<int>(0, calibration_sum.width() / 2 - 10);
  int forward_max_x = std::min<int>(calibration_sum.width() - 1, calibration_sum.width() / 2 + 10);
  int forward_min_y = std::max<int>(0, calibration_sum.height() / 2 - 10);
  int forward_max_y = std::min<int>(calibration_sum.height() - 1, calibration_sum.height() / 2 + 10);
  Vec3d forward_sum = Vec3d::Zero();
  u32 forward_count = 0;
  for (u32 y = forward_min_y; y <= forward_max_y; ++ y) {
    for (u32 x = forward_min_x; x <= forward_max_x; ++ x) {
      forward_sum += calibration_sum(x, y);
      forward_count += calibration_count(x, y);
    }
  }
  
  // Fallback: use whole image
  if (forward_count == 0) {
    forward_min_x = 0;
    forward_max_x = calibration_sum.width() - 1;
    forward_min_y = 0;
    forward_max_y = calibration_sum.height() - 1;
    for (u32 y = forward_min_y; y <= forward_max_y; ++ y) {
      for (u32 x = forward_min_x; x <= forward_max_x; ++ x) {
        forward_sum += calibration_sum(x, y);
        forward_count += calibration_count(x, y);
      }
    }
  }
  
  Mat3d forward_rotation = Mat3d::Identity();
  if (forward_count > 0) {
    Vec3d forward = forward_sum / forward_count;
    forward_rotation = Quaterniond::FromTwoVectors(forward, Vec3d(0, 0, 1)).toRotationMatrix();
  }
  
  const int right_min_x = std::min<int>(calibration_sum.width() - 1, calibration_sum.width() / 2 + 11);
  const int right_max_x = calibration_sum.width() - 1;
  const int right_min_y = std::max<int>(0, calibration_sum.height() / 2 - 10);
  const int right_max_y = std::min<int>(calibration_sum.height() - 1, calibration_sum.height() / 2 + 10);
  Vec3d right_sum = Vec3d::Zero();
  u32 right_count = 0;
  for (u32 y = right_min_y; y <= right_max_y; ++ y) {
    for (u32 x = right_min_x; x <= right_max_x; ++ x) {
      right_sum += calibration_sum(x, y);
      right_count += calibration_count(x, y);
    }
  }
  
  Mat3d right_rotation = Mat3d::Identity();
  if (right_count > 0) {
    Vec3d forward_rotated_right = forward_rotation * (right_sum / right_count);
    
    // We want to rotate forward_rotated_right around the forward vector (0, 0, 1) such as to maximize its x value.
    double angle = atan2(-forward_rotated_right.y(), forward_rotated_right.x());  // TODO: Is the minus here correct?
    right_rotation = AngleAxisd(angle, Vec3d(0, 0, 1)).toRotationMatrix();
  }
  
  Mat3d rotation = right_rotation * forward_rotation;
  for (u32 y = 0; y < calibration_sum.height(); ++ y) {
    for (u32 x = 0; x < calibration_sum.width(); ++ x) {
      if (calibration_count(x, y) > 0) {
        calibration_sum(x, y) = rotation * calibration_sum(x, y);
      }
    }
  }
  
  SE3d rotation_transform(rotation, Vec3d::Zero());
  for (usize i = 0; i < image_used.size(); ++ i) {
    if (!image_used.at(i)) {
      continue;
    }
    
    image_tr_global.at(i) = rotation_transform * image_tr_global.at(i);
  }
}


void DebugDisplaySparseAndDenseMatches(
    Dataset* dataset,
    int camera_index,
    int imageset_index,
    const Image<Vec3d>& dense_matches) {
  static ImageDisplay display;
  display.Clear();
  
  // Visualize dense matches as image
  Image<Vec3u8> cur_dense_matches_visualization(dense_matches.size());
  for (u32 y = 0; y < cur_dense_matches_visualization.height(); ++ y) {
    for (u32 x = 0; x < cur_dense_matches_visualization.width(); ++ x) {
      Vec3d match = dense_matches(x, y);
      if (match.hasNaN()) {
        cur_dense_matches_visualization(x, y) = Vec3u8(0, 0, 0);
      } else {
        cur_dense_matches_visualization(x, y) = Vec3u8(40 * match.x(), 40 * match.y(), 40 * match.z());
      }
    }
  }
  
  // Visualize sparse matches as subpixel elements
  vector<PointFeature>& matches = dataset->GetImageset(imageset_index)->FeaturesOfCamera(camera_index);
//   const KnownGeometry& kg = dataset->GetKnownGeometry(init_known_geometry_index);
  for (const PointFeature& match : matches) {
    Vec2f position = Vec2f(dense_matches.width() / static_cast<float>(dataset->GetImageSize(camera_index).x()),
                           dense_matches.height() / static_cast<float>(dataset->GetImageSize(camera_index).y())).cwiseProduct(match.xy);
    
    display.AddSubpixelDotPixelCornerConv(position, Vec3u8(255, 255, 255));
    
    // Add the coordinates of the feature in the pattern as text
//     auto it = kg.feature_id_to_position.find(match.id);
//     if (it != kg.feature_id_to_position.end()) {
//       const auto& pattern_position = it->second;
//       std::ostringstream text;
//       text << pattern_position.x() << ", " << pattern_position.y();
//       display.AddSubpixelTextPixelCornerConv(position, Vec3u8(255, 255, 255), text.str());
//     }
  }
  
  display.Update(cur_dense_matches_visualization, "Sparse and densified matches");
}


void VisualizeCurrentState(
    Dataset* dataset,
    int camera_index,
    const vector<bool>& image_used,
    const Image<Vec3d>& calibration_sum,
    const vector<bool>& known_geometry_localized,
    const vector<Mat3f>& global_r_known_geometry,
    const vector<Vec3f>& global_t_known_geometry) {
  // Find the first localized image to start the visualization with
  int visualized_imageset = -1;
  for (usize i = 0; i < image_used.size(); ++ i) {
    if (image_used[i]) {
      visualized_imageset = i;
      break;
    }
  }
  if (visualized_imageset < 0) {
    LOG(ERROR) << "Could not find any localized image to visualize.";
    return;
  }
  
  // Show the visualization and react to keypresses.
  bool update_visualization = true;
  while (true) {
    if (update_visualization) {
      LOG(INFO) << "Visualizing imageset " << visualized_imageset << ".";
      
      Image<Vec3d> dense_matches;
      vector<PointFeature>& matches = dataset->GetImageset(visualized_imageset)->FeaturesOfCamera(camera_index);
      int match_count = DensifyMatches(
          matches,
          dataset,
          camera_index,
          known_geometry_localized,
          global_r_known_geometry,
          global_t_known_geometry,
          calibration_sum.width(),
          calibration_sum.height(),
          &dense_matches);
      LOG(1) << "Dense match count: " << match_count;
      DebugDisplaySparseAndDenseMatches(
          dataset,
          camera_index,
          visualized_imageset,
          dense_matches);
      
      LOG(1) << "  Press Return to exit the visualization.";
      LOG(1) << "  Press A to go to the previous imageset.";
      LOG(1) << "  Press D to go to the next imageset.";
      
      update_visualization = false;
    }
    
    // React to keypresses
    constexpr int kReturnKeyCode = 10;
    int key = GetKeyInput();
    
    if (key == kReturnKeyCode) {
      break;
    } else if (key == 'a') {
      // Go to previous imageset.
      int old_visualized_imageset = visualized_imageset;
      -- visualized_imageset;
      while (visualized_imageset >= 0 &&
             !image_used[visualized_imageset]) {
        -- visualized_imageset;
      }
      if (visualized_imageset >= 0) {
        update_visualization = true;
      } else {
        LOG(WARNING) << "No previous localized imageset exists.";
        visualized_imageset = old_visualized_imageset;
      }
    } else if (key == 'd') {
      // Go to next imageset.
      int old_visualized_imageset = visualized_imageset;
      ++ visualized_imageset;
      while (visualized_imageset < image_used.size() &&
             !image_used[visualized_imageset]) {
        ++ visualized_imageset;
      }
      if (visualized_imageset < image_used.size()) {
        update_visualization = true;
      } else {
        LOG(WARNING) << "No next localized imageset exists.";
        visualized_imageset = old_visualized_imageset;
      }
    }
  }
}


void DenseInitialization::VisualizeIntrinsics(
    const DenseInitializationParameters& p,
    const Image<Vec3d>& dense_intrinsics,
    Image<u32>* calibration_count,
    const vector<bool>& camera_image_used) {
  Image<Vec3u8> visualization(dense_intrinsics.size());
  
  for (u32 y = 0; y < visualization.height(); ++ y) {
    for (u32 x = 0; x < visualization.width(); ++ x) {
      if (calibration_count && calibration_count->at(x, y) == 0) {
        visualization(x, y) = Vec3u8(0, 0, 0);
      } else {
        Vec3d direction = dense_intrinsics(x, y).normalized();
        visualization(x, y) = Vec3u8(
            70 * 255.99f / 2.f * (direction.x() + 1),
            70 * 255.99f / 2.f * (direction.y() + 1),
            270 * 255.99f / 2.f * (direction.z() + 1));
      }
    }
  }
  
  p.calibration_window->UpdateInitialization(p.camera_index, visualization);
  
  if (p.step_by_step) {
    LOG(INFO) << "> Press Return to continue (v to visualize the current state)";
    
    constexpr int kReturnKeyCode = 10;
    while (true) {
      int key = GetKeyInput();
      if (key == 'v') {
        VisualizeCurrentState(
            p.dataset, p.camera_index, camera_image_used, dense_intrinsics,
            known_geometry_localized, global_r_known_geometry, global_t_known_geometry);
      } else if (key == kReturnKeyCode) {
        break;
      }
    }
  }
}


bool DenseInitialization::AttemptRelativePoseInitialization(
    const DenseInitializationParameters& p,
    int init_indices[3],
    Image<Vec3d> dense_matches[3],
    SE3d cloud2_tr_cloud[2],
    Vec3d* optical_center,
    int* init_known_geometry_index,
    usize* num_point_triples) {
  // If no camera has been calibrated before, choose known_geometry_index as
  // the KnownGeometry where we have the most matches for and discard the others,
  // since we cannot use different geometries with unknown relative poses yet.
  // If a camera has been calibrated before, make sure to use a known_geometry_index
  // that has already been localized (otherwise, we risk potentially creating
  // two separate reconstructions that do not have any overlap). In this case,
  // we could theoretically even use more than one geometry if we know their
  // relative poses already, but this is not implemented yet (TODO).
  // LOG(1) << "Choosing known geometry with highest min_match_count ...";
  
  vector<int> min_match_count_for_geometry(p.dataset->KnownGeometriesCount(), numeric_limits<int>::max());
  for (int image = 0; image < 3; ++ image) {
    vector<int> match_count_for_geometry(p.dataset->KnownGeometriesCount(), 0);
    
    const vector<PointFeature>& matches = p.dataset->GetImageset(init_indices[image])->FeaturesOfCamera(p.camera_index);
    for (usize m = 0; m < matches.size(); ++ m) {
      const PointFeature& match = matches[m];
      
      for (int k = 0; k < p.dataset->KnownGeometriesCount(); ++ k) {
        if (known_geometry_localized.size() > k && !known_geometry_localized[k]) {
          continue;
        }
        
        const KnownGeometry& kg = p.dataset->GetKnownGeometry(k);
        auto it = kg.feature_id_to_position.find(match.id);
        if (it != kg.feature_id_to_position.end()) {
          ++ match_count_for_geometry[k];
        }
      }
    }
    
    for (int k = 0; k < p.dataset->KnownGeometriesCount(); ++ k) {
      // LOG(1) << "match_count_for_geometry[" << k << "]: " << match_count_for_geometry[k];
      min_match_count_for_geometry[k] = std::min(min_match_count_for_geometry[k], match_count_for_geometry[k]);
    }
  }
  
  *init_known_geometry_index = 0;
  for (int k = 0; k < p.dataset->KnownGeometriesCount(); ++ k) {
    if (min_match_count_for_geometry[k] > min_match_count_for_geometry[*init_known_geometry_index]) {
      *init_known_geometry_index = k;
    }
  }
  
  if (min_match_count_for_geometry[*init_known_geometry_index] < 20) {  // TODO: Threshold is somewhat arbitrary
    // LOG(INFO) << "Calibration failed: min_match_count_for_geometry[init_known_geometry_index] == " << min_match_count_for_geometry[*init_known_geometry_index] << " < 20";
    return false;
  }
  
  vector<bool> cur_known_geometry_localized(p.dataset->KnownGeometriesCount(), false);
  vector<Mat3f> cur_global_r_known_geometry(p.dataset->KnownGeometriesCount());
  vector<Vec3f> cur_global_t_known_geometry(p.dataset->KnownGeometriesCount());
  
  cur_known_geometry_localized[*init_known_geometry_index] = true;
  cur_global_r_known_geometry[*init_known_geometry_index] = Mat3f::Identity();
  cur_global_t_known_geometry[*init_known_geometry_index] = Vec3f::Zero();
  
  // Estimate the initial calibration on a small image size for higher speed
  // since high spatial accuracy is likely not neccesary here
  int max_init_pixel_count = 640 * 480;  // TODO: Somewhat arbitrary setting
  int actual_pixel_count = p.dataset->GetImageSize(p.camera_index).x() * p.dataset->GetImageSize(p.camera_index).y();
  float scaling = std::min<float>(1, sqrt(max_init_pixel_count / static_cast<float>(actual_pixel_count)));
  const int buffer_width = scaling * p.dataset->GetImageSize(p.camera_index).x();
  const int buffer_height = scaling * p.dataset->GetImageSize(p.camera_index).y();
  
  // Interpolate matches within the checkerboard squares to get dense matches (only used for initialization)
  // LOG(1) << "Densify matches in initialization images ...";
  for (int i = 0; i < 3; ++ i) {
    vector<PointFeature>& matches = p.dataset->GetImageset(init_indices[i])->FeaturesOfCamera(p.camera_index);
    DensifyMatches(matches,
                    p.dataset,
                    p.camera_index,
                    cur_known_geometry_localized,
                    cur_global_r_known_geometry,
                    cur_global_t_known_geometry,
                    buffer_width,
                    buffer_height,
                    &dense_matches[i]);
  }
  
  // DEBUG: Visualize densified matches
  // if (show_visualizations && step_by_step) {
  //   for (int i = 0; i < 3; ++ i) {
  //     DebugDisplaySparseAndDenseMatches(
  //         dataset,
  //         camera_index,
  //         init_indices[i],
  //         dense_matches[i]);
  //     if (step_by_step) { LOG(INFO) << "> Press Return to continue"; std::getchar(); }
  //   }
  // }
  
  // Attempt initialization: Compute relative poses of the calibration target in the three images.
  // Use all pixels which have matches in all three images.
  // LOG(1) << "Running relative pose initializer ...";
  
  *num_point_triples = 0;
  for (u32 y = 0; y < dense_matches[0].height(); ++ y) {
    for (u32 x = 0; x < dense_matches[0].width(); ++ x) {
      if (!dense_matches[0](x, y).hasNaN() &&
          !dense_matches[1](x, y).hasNaN() &&
          !dense_matches[2](x, y).hasNaN()) {
        ++ (*num_point_triples);
      }
    }
  }
  
  // LOG(INFO) << "Image fraction covered by point triples: " << ((100. * (*num_point_triples)) / dense_matches[0].pixel_count());
  // TODO: This threshold is somewhat arbitrary. This is chosen higher than the minimum point triple count since we operate on interpolated matches here.
  constexpr double kMinRequiredMatchedImageAreaFraction = 0.01; // 0.05;  // TODO: Make configurable?
  if (*num_point_triples < kMinRequiredMatchedImageAreaFraction * dense_matches[0].pixel_count()) {
    // LOG(INFO) << "Calibration failed: num_point_triples == " << *num_point_triples << " < " << (kMinRequiredMatchedImageAreaFraction * dense_matches[0].pixel_count());
    return false;
  }
  
  Point3fCloud initialization_clouds[3];
  for (int i = 0; i < 3; ++ i) {
    initialization_clouds[i].Resize(*num_point_triples);
  }
  int triple_index = 0;
  for (u32 y = 0; y < dense_matches[0].height(); ++ y) {
    for (u32 x = 0; x < dense_matches[0].width(); ++ x) {
      Vec3d m0 = dense_matches[0](x, y);
      Vec3d m1 = dense_matches[1](x, y);
      Vec3d m2 = dense_matches[2](x, y);
      if (!m0.hasNaN() &&
          !m1.hasNaN() &&
          !m2.hasNaN()) {
        initialization_clouds[0].at(triple_index).position() = m0.cast<float>();
        initialization_clouds[1].at(triple_index).position() = m1.cast<float>();
        initialization_clouds[2].at(triple_index).position() = m2.cast<float>();
        
        ++ triple_index;
      }
    }
  }
  
  if (!CentralCameraPlanarCalibrationObjectRelativePose(initialization_clouds, cloud2_tr_cloud, optical_center)) {
    // LOG(INFO) << "Calibration failed: Relative pose initializer returned false.";
    return false;
  }
  
  // TODO: In principle, would have to check for degenerate conditions for the
  //       relative pose initializer instead of always assuming success here.
  return true;
}


void DenseInitialization::InitializeForLocalization(
    const DenseInitializationParameters& p,
    Image<Vec3d>* calibration_sum,
    Image<u32>* calibration_count) {
  // Initialize any known geometry to identity.
  // TODO: Make sure that it is actually observed by the current camera.
  int init_known_geometry_index = 0;
  if (known_geometry_localized.empty()) {
    known_geometry_localized.resize(p.dataset->KnownGeometriesCount(), false);
    global_r_known_geometry.resize(p.dataset->KnownGeometriesCount());
    global_t_known_geometry.resize(p.dataset->KnownGeometriesCount());
    
    known_geometry_localized[init_known_geometry_index] = true;
    global_r_known_geometry[init_known_geometry_index] = Mat3f::Identity();
    global_t_known_geometry[init_known_geometry_index] = Vec3f::Zero();
  }
  
  image_used.resize(p.dataset->num_cameras());
  vector<bool>& camera_image_used = image_used[p.camera_index];
  camera_image_used.resize(p.dataset->ImagesetCount(), false);
  
  image_tr_global.resize(p.dataset->num_cameras());
  vector<SE3d>& camera_image_tr_global = image_tr_global[p.camera_index];
  camera_image_tr_global.resize(p.dataset->ImagesetCount());
  
  *calibration_sum = observation_directions[p.camera_index];
  calibration_count->SetSize(calibration_sum->size());
  for (int y = 0; y < calibration_count->height(); ++ y) {
    for (int x = 0; x < calibration_count->width(); ++ x) {
      if ((*calibration_sum)(x, y).hasNaN()) {
        (*calibration_count)(x, y) = 0;
      } else {
        (*calibration_count)(x, y) = 1;
      }
    }
  }
}


void DenseInitialization::InitializeFromRelativePoses(
    const DenseInitializationParameters& p,
    int init_indices[3],
    int init_known_geometry_index,
    Image<Vec3d> dense_matches[3],
    SE3d cloud2_tr_cloud[2],
    const Vec3d& optical_center,
    Image<Vec3d>* calibration_sum,
    Image<u32>* calibration_count) {
  // Initialize known_geometry_localized, global_r_known_geometry, and global_t_known_geometry
  // (if it has not been initialized before by another camera already).
  if (known_geometry_localized.empty()) {
    known_geometry_localized.resize(p.dataset->KnownGeometriesCount(), false);
    global_r_known_geometry.resize(p.dataset->KnownGeometriesCount());
    global_t_known_geometry.resize(p.dataset->KnownGeometriesCount());
    
    known_geometry_localized[init_known_geometry_index] = true;
    global_r_known_geometry[init_known_geometry_index] = Mat3f::Identity();
    global_t_known_geometry[init_known_geometry_index] = Vec3f::Zero();
  }
  
  // Initialize the camera pose (relative to initialization_clouds[2], i.e., the pattern used for initialization, in identity pose).
  // The viewing direction can be chosen freely: it does not matter since it can be compensated by a rotation of the intrinsics.
  // So, simply choose identity rotation here.
  SE3d cloud2_tr_camera(Mat3d::Identity(), optical_center.cast<double>());
  LOG(INFO) << "optical_center: " << optical_center.transpose();
  LOG(INFO) << "init_known_geometry_index: " << init_known_geometry_index;
  
  // Using the relative target poses, initialize the intrinsics in image areas covered in at least one of the three images.
  // This is approximate since we still use the interpolated matches.
  LOG(INFO) << "Determining initial partial calibration from initial images ...";
  
  calibration_sum->SetSize(dense_matches[0].size());
  calibration_sum->SetTo(Vec3d::Zero());
  calibration_count->SetSize(dense_matches[0].size());
  calibration_count->SetTo(static_cast<u32>(0));
  
  for (int i = 0; i < 3; ++ i) {
    Mat3d R;
    Vec3d t;
    SE3d camera_tr_cloud;
    if (i < 2) {
      camera_tr_cloud = cloud2_tr_camera.inverse() * cloud2_tr_cloud[i];
    } else {
      camera_tr_cloud = cloud2_tr_camera.inverse();
    }
    R = camera_tr_cloud.rotationMatrix();
    t = camera_tr_cloud.translation();
    
    for (u32 y = 0; y < calibration_sum->height(); ++ y) {
      for (u32 x = 0; x < calibration_sum->width(); ++ x) {
        Vec3d match = dense_matches[i](x, y);
        if (match.hasNaN()) {
          continue;
        }
        
        // Transform the point on the pattern into camera space.
        Vec3d point = R * match + t;
        
        calibration_sum->at(x, y) += point.normalized();
        calibration_count->at(x, y) += 1;
      }
    }
  }
  
  // Look at the remaining images to try to initialize the remaining image areas.
  // We can localize the images using the already initialized image region, and then use them to initialize other image
  // regions that they cover.
  LOG(INFO) << "Extending initial calibration by localizing the remaining images ...";
  
  image_used.resize(p.dataset->num_cameras());
  vector<bool>& camera_image_used = image_used[p.camera_index];
  camera_image_used.resize(p.dataset->ImagesetCount(), false);
  for (int i = 0; i < 3; ++ i) {
    camera_image_used[init_indices[i]] = true;
  }
  
  image_tr_global.resize(p.dataset->num_cameras());
  vector<SE3d>& camera_image_tr_global = image_tr_global[p.camera_index];
  camera_image_tr_global.resize(p.dataset->ImagesetCount());
  SE3d camera_tr_cloud2 = cloud2_tr_camera.inverse();
  SE3d global_tr_known_geometry = SE3d(global_r_known_geometry[init_known_geometry_index].cast<double>(),
                                       global_t_known_geometry[init_known_geometry_index].cast<double>());
  SE3d known_geometry_tr_global = global_tr_known_geometry.inverse();
  camera_image_tr_global[init_indices[0]] = camera_tr_cloud2 * cloud2_tr_cloud[0] * known_geometry_tr_global;
  camera_image_tr_global[init_indices[1]] = camera_tr_cloud2 * cloud2_tr_cloud[1] * known_geometry_tr_global;
  camera_image_tr_global[init_indices[2]] = camera_tr_cloud2 * known_geometry_tr_global;
  
  LocalizeAdditionalPatterns(
      p, init_indices[0], camera_image_tr_global[init_indices[0]].inverse(),
      *calibration_sum, *calibration_count);
  LocalizeAdditionalPatterns(
      p, init_indices[1], camera_image_tr_global[init_indices[1]].inverse(),
      *calibration_sum, *calibration_count);
  LocalizeAdditionalPatterns(
      p, init_indices[2], camera_image_tr_global[init_indices[2]].inverse(),
      *calibration_sum, *calibration_count);
}


bool DenseInitialization::AttemptToLocalizeImage(
    const DenseInitializationParameters& p,
    int imageset_index,
    const Image<Vec3d>& calibration_sum,
    const Image<u32>& calibration_count,
    SE3d* global_tr_image,
    Image<Vec3d>* dense_matches) {
  vector<PointFeature>& matches = p.dataset->GetImageset(imageset_index)->FeaturesOfCamera(p.camera_index);
  
  // Try to localize the image using sparse matches first.
  double scale_x = static_cast<double>(calibration_sum.width()) / p.dataset->GetImageSize(p.camera_index).x();
  double scale_y = static_cast<double>(calibration_sum.height()) / p.dataset->GetImageSize(p.camera_index).y();
  vector<Vec3d> pattern;
  vector<Vec2d> features;
  for (usize m = 0; m < matches.size(); ++ m) {
    const PointFeature& match = matches[m];
    
    for (int k = 0; k < p.dataset->KnownGeometriesCount(); ++ k) {
      if (!known_geometry_localized[k]) {
        continue;
      }
      const KnownGeometry& kg = p.dataset->GetKnownGeometry(k);
      auto it = kg.feature_id_to_position.find(match.id);
      if (it != kg.feature_id_to_position.end()) {
        Vec3f pattern_point = Vec3f(kg.cell_length_in_meters * it->second.x(), kg.cell_length_in_meters * it->second.y(), 0);
        Vec3f global_point = global_r_known_geometry[k] * pattern_point + global_t_known_geometry[k];
        pattern.push_back(global_point.cast<double>());
        features.push_back(Vec2f(scale_x, scale_y).cwiseProduct(match.xy).cast<double>());
        break;
      }
    }
  }
  bool image_localized = false;
  if (pattern.size() > 7) {  // TODO: Somewhat arbitrary threshold, make configurable?
    image_localized = LocalizePattern(
        calibration_sum,
        calibration_count,
        pattern,
        features,
        0,
        features.size() - 1,
        7,  // TODO: min_calibrated_match_count, make configurable?
        global_tr_image);
  }
  pattern.clear();
  features.clear();
  
  // Get and densify the matches.
  // TODO: Only compute this once and then cache it
  int match_count = DensifyMatches(
      matches,
      p.dataset,
      p.camera_index,
      known_geometry_localized,
      global_r_known_geometry,
      global_t_known_geometry,
      calibration_sum.width(),
      calibration_sum.height(),
      dense_matches);
  
  // Attempt to localize the image using the dense matches if it did not work with the sparse ones.
  if (!image_localized) {
    pattern.reserve(match_count);
    features.reserve(match_count);
    
    for (u32 y = 0; y < dense_matches->height(); ++ y) {
      for (u32 x = 0; x < dense_matches->width(); ++ x) {
        Vec3d match = dense_matches->at(x, y);
        if (std::isnan(match.x())) {
          continue;
        }
        
        pattern.push_back(match);
        features.push_back(Vec2d(x + 0.5f, y + 0.5f));
      }
    }
    
    if (features.size() < 50) {
      return false;
    }
    
    if (!LocalizePattern(
        calibration_sum,
        calibration_count,
        pattern,
        features,
        0,
        features.size() - 1,
        50,  // TODO: min_calibrated_match_count, make configurable?
        global_tr_image)) {
      return false;
    }
    image_localized = true;  // NOTE: not used below
  }
  
  return true;
}


void DenseInitialization::UpdateCalibrationWithImage(
    const SE3d& camera_tr_global,
    const Image<Vec3d>& dense_matches,
    Image<Vec3d>* calibration_sum,
    Image<u32>* calibration_count) {
  Mat3d R = camera_tr_global.rotationMatrix().cast<double>();
  Vec3d t = camera_tr_global.translation().cast<double>();
  
  for (u32 y = 0; y < calibration_sum->height(); ++ y) {
    for (u32 x = 0; x < calibration_sum->width(); ++ x) {
      Vec3d match = dense_matches(x, y);
      if (std::isnan(match.x())) {
        continue;
      }
      
      // Transform the point from global to camera space.
      Vec3d camera_point = R * match + t;
      calibration_sum->at(x, y) += camera_point.normalized();
      calibration_count->at(x, y) += 1;
    }
  }
}

bool DenseInitialization::MakeNewSubmodelForKnownGeometry(
    int known_geometry_index) {
  LOG(INFO) << "Since no images link known geometry " << known_geometry_index << " with the existing part of the model, making a new sub-model for it.";
  
  // We simply pretend that the known geometry has been localized and is at
  // identity. Subsequent attempts to localize images will then succeed against it.
  known_geometry_localized[known_geometry_index] = true;
  global_r_known_geometry[known_geometry_index] = Mat3f::Identity();
  global_t_known_geometry[known_geometry_index] = Vec3f::Zero();
  
  return true;
}


void ChooseRandomInitIndices(
    Dataset* dataset,
    int init_indices[3]) {
  // The image indices should be chosen such that there are many matches to a
  // single common KnownGeometry in all images, with large overlap. However, the
  // image poses must not be exactly the same.
  // TODO: Can we do better choices than fully random?
  init_indices[0] = rand() % dataset->ImagesetCount();
  
  init_indices[1] = rand() % (dataset->ImagesetCount() - 1);
  if (init_indices[1] >= init_indices[0]) {
    ++ init_indices[1];
  }
  CHECK_LT(init_indices[1], dataset->ImagesetCount());
  
  init_indices[2] = rand() % (dataset->ImagesetCount() - 2);
  if (init_indices[2] >= std::min(init_indices[0], init_indices[1])) {
    ++ init_indices[2];
  }
  if (init_indices[2] >= std::max(init_indices[0], init_indices[1])) {
    ++ init_indices[2];
  }
  CHECK_LT(init_indices[2], dataset->ImagesetCount());
  
  // for (int i = 0; i < 3; ++ i) {
  //   LOG(INFO) << "init_indices[" << i << "]: " << init_indices[i];
  // }
}


bool DenseInitialization::InitializeCamera(
    Dataset* dataset,
    int camera_index,
    bool localize_only,
    CalibrationWindow* calibration_window,
    bool step_by_step) {
  srand(time(nullptr));  // TODO: Leave that to the caller such that a specific state can be set if desired?
  
  // Group together commonly used parameters for convenience.
  DenseInitializationParameters p;
  p.dataset = dataset;
  p.camera_index = camera_index;
  p.calibration_window = calibration_window;
  p.step_by_step = step_by_step;
  
  // Perform a number of initialization attempts and pick the result which seems
  // to be best.
  int init_indices[3];
  Image<Vec3d> dense_matches[3];
  SE3d cloud2_tr_cloud[2];
  Vec3d optical_center;
  
  int best_num_point_triples = 0;
  int init_known_geometry_index = -1;
  
  const int kMaxNumAttempts = localize_only ? 0 : 500;
  for (int attempt = 0; attempt < kMaxNumAttempts; ++ attempt) {
    // Choose three images for initialization.
    int cur_init_indices[3];
    ChooseRandomInitIndices(dataset, cur_init_indices);
    
    // Attempt relative pose initialization with the chosen image triple.
    Image<Vec3d> cur_dense_matches[3];
    SE3d cur_cloud2_tr_cloud[2];
    Vec3d cur_optical_center;
    int cur_init_known_geometry_index;
    usize num_point_triples;
    if (!AttemptRelativePoseInitialization(
        p, cur_init_indices, cur_dense_matches, cur_cloud2_tr_cloud,
        &cur_optical_center, &cur_init_known_geometry_index, &num_point_triples)) {
      continue;
    }
    
    // If this was the best calibration attempt so far, remember it.
    if (num_point_triples > best_num_point_triples) {
      best_num_point_triples = num_point_triples;
      
      for (int i = 0; i < 3; ++ i) {
        init_indices[i] = cur_init_indices[i];
        dense_matches[i] = cur_dense_matches[i];
      }
      for (int i = 0; i < 2; ++ i) {
        cloud2_tr_cloud[i] = cur_cloud2_tr_cloud[i];
      }
      optical_center = cur_optical_center;
      init_known_geometry_index = cur_init_known_geometry_index;
      
      // Immediately accept this calibration if it seems very good.
      constexpr double kMatchedImageAreaFractionForImmediateAcceptance = 0.3;
      if (num_point_triples >= kMatchedImageAreaFractionForImmediateAcceptance * cur_dense_matches[0].pixel_count()) {
        LOG(INFO) << "Accepting the current calibration immediately.";
        break;
      }
    }
  }
  
  if (!localize_only && best_num_point_triples == 0) {
    LOG(ERROR) << "Exceeded the number of failed initialization attempts. Giving up.";
    return false;
  }
  
  if (!localize_only) {
    LOG(INFO) << "Final init_indices: " << init_indices[0] << ", " << init_indices[1] << ", " << init_indices[2];
    LOG(INFO) << "best_num_point_triples: " << best_num_point_triples;
  }
  
  // Given the relative poses between three images, initialize the remaining
  // state and an initial dense calibration.
  Image<Vec3d> calibration_sum;
  Image<u32> calibration_count;
  if (localize_only) {
    InitializeForLocalization(p, &calibration_sum, &calibration_count);
  } else {
    InitializeFromRelativePoses(
        p, init_indices, init_known_geometry_index, dense_matches,
        cloud2_tr_cloud, optical_center, &calibration_sum, &calibration_count);
  }
  
  vector<bool>& camera_image_used = image_used[camera_index];
  vector<SE3d>& camera_image_tr_global = image_tr_global[camera_index];
  
  if (p.calibration_window) {
    VisualizeIntrinsics(p, calibration_sum, &calibration_count, camera_image_used);
  }
  
  // Attempt to localize all remaining images and extend the dense calibration.
  while (true) {
    int num_imagesets_localized = 0;
    for (usize i = 0; i < camera_image_used.size(); ++ i) {
      if (camera_image_used[i]) {
        ++ num_imagesets_localized;
      }
    }
    
    while (true) {
      int old_num_imagesets_localized = num_imagesets_localized;
      
      for (int imageset_index = 0; imageset_index < dataset->ImagesetCount(); ++ imageset_index) {
        if (camera_image_used[imageset_index]) {
          continue;
        }
        
        SE3d global_tr_image;
        Image<Vec3d> dense_matches;
        if (!AttemptToLocalizeImage(p, imageset_index, calibration_sum, calibration_count, &global_tr_image, &dense_matches)) {
          continue;
        }
        
        // Succeeded to localize this image. Update the estimated calibration.
        LOG(INFO) << "Adding image " << imageset_index;
        
        SE3d camera_tr_global = global_tr_image.inverse();
        camera_image_tr_global[imageset_index] = camera_tr_global;
        camera_image_used[imageset_index] = true;
        
        if (!localize_only) {
          UpdateCalibrationWithImage(
              camera_tr_global, dense_matches, &calibration_sum, &calibration_count);
        }
        
        LocalizeAdditionalPatterns(
            p, imageset_index, camera_image_tr_global[imageset_index].inverse(),
            calibration_sum, calibration_count);
        
        ++ num_imagesets_localized;
        
        // TODO: Triggering this alternating BA should probably also be related to
        //       achieving certain completenesses in image coverage.
        if (!localize_only && num_imagesets_localized < 50 && num_imagesets_localized % 10 == 0) {
          AlternatingBundleAdjustment(p, calibration_sum, calibration_count, /*max_iterations*/ 5);
        }
        
        if (!localize_only && ((num_imagesets_localized < 100 && num_imagesets_localized % 10 == 0) ||
                               (num_imagesets_localized % 100 == 0))) {
          ChooseNiceCameraOrientation(
              calibration_sum,
              calibration_count,
              camera_image_used,
              camera_image_tr_global);
        }
        
        if (!localize_only && p.calibration_window) {
          VisualizeIntrinsics(p, calibration_sum, &calibration_count, camera_image_used);
        }
      }
      
      if (num_imagesets_localized == old_num_imagesets_localized) {
        break;
      }
    }
    
    // If there are any known geometries which are not localized yet, make a new
    // submodel for one of them and repeat trying to add images.
    // TODO: This must be done after looking at all cameras, not within the
    //       handling of a single camera.
    bool localized_new_kg = false;
    for (int k = 0; k < known_geometry_localized.size(); ++ k) {
      if (!known_geometry_localized[k] && MakeNewSubmodelForKnownGeometry(k)) {
        localized_new_kg = true;
        break;
      }
    }
    
    if (!localized_new_kg) {
      break;
    }
  }
  
  // Output the amount of imagesets used
  int num_images_used = 0;
  for (int imageset_index = 0; imageset_index < dataset->ImagesetCount(); ++ imageset_index) {
    if (camera_image_used[imageset_index]) {
      ++ num_images_used;
    }
  }
  LOG(INFO) << "Imagesets used for " << (localize_only ? "localization" : "calibration") << ": " << num_images_used << " / " << dataset->ImagesetCount();
  
  // Beautify the final camera orientation
  if (!localize_only) {
    ChooseNiceCameraOrientation(
        calibration_sum,
        calibration_count,
        camera_image_used,
        camera_image_tr_global);
  }
  
  // Create the initial dense model by normalizing calibration_sum.
  if (!localize_only) {
    if (observation_directions.size() <= camera_index) {
      observation_directions.resize(camera_index + 1);
    }
    Image<Vec3d>& intrinsics = observation_directions[camera_index];
    intrinsics.SetSize(calibration_sum.size());
    for (u32 y = 0; y < intrinsics.height(); ++ y) {
      for (u32 x = 0; x < intrinsics.width(); ++ x) {
        if (calibration_count(x, y) == 0) {
          intrinsics(x, y) = Vec3d::Constant(numeric_limits<double>::quiet_NaN());
        } else {
          intrinsics(x, y) = calibration_sum(x, y).normalized();
        }
      }
    }
    
    if (p.calibration_window) {
      VisualizeIntrinsics(p, intrinsics, &calibration_count, camera_image_used);
    }
  }
  
  return true;
}

}
