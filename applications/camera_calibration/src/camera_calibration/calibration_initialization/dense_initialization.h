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

#include <vector>

#include <libvis/eigen.h>
#include <libvis/image.h>
#include <libvis/libvis.h>
#include <libvis/sophus.h>

namespace vis {

class CalibrationWindow;
class Dataset;
class ImageDisplay;
struct DenseInitializationParameters;

/// Groups variables that constitute the dense initialization.
/// TODO: This only applies to central cameras; we should also be able to initialize non-central cameras directly.
struct DenseInitialization {
  /// Tries to obtain a dense calibration initialization for the given camera
  /// index. If localize_only is true, a dense model for the camera must be in
  /// observation_directions[camera_index] already.
  bool InitializeCamera(
      Dataset* dataset,
      int camera_index,
      bool localize_only,
      CalibrationWindow* calibration_window,
      bool step_by_step);
  
  inline int num_cameras() const { return image_used.size(); }
  inline int num_imagesets() const { return image_used[0].size(); }
  
  
  /// Geometry properties, indexed by: [known_geometry_index].
  vector<bool> known_geometry_localized;
  vector<Mat3f> global_r_known_geometry;
  vector<Vec3f> global_t_known_geometry;
  
  /// Image properties, indexed by: [camera_index][imageset_index].
  vector<vector<bool>> image_used;
  vector<vector<SE3d>> image_tr_global;
  
  /// For each pixel, specifies the observation direction.
  /// Indexed by: [camera_index].
  vector<Image<Vec3d>> observation_directions;
  
  
 private:
  /// Attempts to initialize the relative poses of the images with the given
  /// init_indices, as well as obtain dense matches and the optical center.
  /// Returns true if successful. In this case, dense_matches, cloud2_tr_cloud,
  /// optical_center, init_known_geometry_index, and num_point_triples will be
  /// set to the results. The chosen known_geometry (returned by
  /// init_known_geometry_index) is assumed to have identity global pose.
  /// The returned optical center position is in cloud2's frame.
  bool AttemptRelativePoseInitialization(
      const DenseInitializationParameters& p,
      int init_indices[3],
      Image<Vec3d> dense_matches[3],
      SE3d cloud2_tr_cloud[2],
      Vec3d* optical_center,
      int* init_known_geometry_index,
      usize* num_point_triples);
  
  /// This is called instead of InitializeFromRelativePoses() if localize_only
  /// is true.
  void InitializeForLocalization(
      const DenseInitializationParameters& p,
      Image<Vec3d>* calibration_sum,
      Image<u32>* calibration_count);
  
  /// Given the relative poses between three images (obtained in
  /// AttemptRelativePoseInitialization()), initializes the remaining state
  /// variables and initial dense intrinsics.
  void InitializeFromRelativePoses(
      const DenseInitializationParameters& p,
      int init_indices[3],
      int init_known_geometry_index,
      Image<Vec3d> dense_matches[3],
      SE3d cloud2_tr_cloud[2],
      const Vec3d& optical_center,
      Image<Vec3d>* calibration_sum,
      Image<u32>* calibration_count);
  
  /// Given the current calibration, attempts to localize the image with the
  /// given index.
  bool AttemptToLocalizeImage(
      const DenseInitializationParameters& p,
      int imageset_index,
      const Image<Vec3d>& calibration_sum,
      const Image<u32>& calibration_count,
      SE3d* global_tr_image,
      Image<Vec3d>* dense_matches);
  
  /// Updates the dense intrinsics with the pattern projection in a newly
  /// localized image.
  void UpdateCalibrationWithImage(
      const SE3d& camera_tr_global,
      const Image<Vec3d>& dense_matches,
      Image<Vec3d>* calibration_sum,
      Image<u32>* calibration_count);
  
  void LocalizeAdditionalPatterns(
      const DenseInitializationParameters& p,
      int imageset_index,
      const SE3d& global_tr_image,
      const Image<Vec3d>& calibration_sum,
      const Image<u32>& calibration_count);
  
  void AlternatingBundleAdjustment(
      const DenseInitializationParameters& p,
      Image<Vec3d>& calibration_sum,
      Image<u32>& calibration_count,
      int max_iterations);
  
  bool MakeNewSubmodelForKnownGeometry(
      int known_geometry_index);
  
  void VisualizeIntrinsics(
      const DenseInitializationParameters& p,
      const Image<Vec3d>& dense_intrinsics,
      Image<u32>* calibration_count,
      const vector<bool>& camera_image_used);
};

}
