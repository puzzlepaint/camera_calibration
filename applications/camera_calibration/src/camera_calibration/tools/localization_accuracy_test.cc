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

#include "camera_calibration/tools/tools.h"

#include <libvis/logging.h>
#include <libvis/statistics.h>
#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac/SampleConsensusProblem.hpp>
#include <opengv/sac/SampleConsensus.hpp>
#include <opengv/sac_problems/point_cloud/PointCloudSacProblem.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>

#include "camera_calibration/fitting_report.h"
#include "camera_calibration/io/calibration_io.h"
#include "camera_calibration/models/central_generic.h"

namespace vis {

int LocalizationAccuracyTest(
    const char* gt_model_yaml_path,
    const char* compared_model_yaml_path) {
  shared_ptr<CameraModel> gt_model = LoadCameraModel(gt_model_yaml_path);
  if (!gt_model) {
    LOG(ERROR) << "Cannot load ground truth camera model: " << gt_model_yaml_path;
    return EXIT_FAILURE;
  }
  
  shared_ptr<CameraModel> compared_model = LoadCameraModel(compared_model_yaml_path);
  if (!compared_model) {
    LOG(ERROR) << "Cannot load camera model to compare: " << compared_model_yaml_path;
    return EXIT_FAILURE;
  }
  
  if (gt_model->width() != compared_model->width() ||
      gt_model->height() != compared_model->height()) {
    LOG(ERROR) << "The ground truth and compared camera models do not have the same image size.";
    return EXIT_FAILURE;
  }
  
  Mean<float> error_distance_mean;
  
  srand(time(nullptr));
  
  constexpr int kNumTrials = 10000;
  vector<float> errors(kNumTrials);
  for (int trial = 0; trial < kNumTrials; ++ trial) {
    // Generate some random points in the image, unproject with the generic camera
    // model to random depths close to 2 meters.
    constexpr int kPointCount = 15;
    constexpr float kMinDistance = 1.5f;
    constexpr float kMaxDistance = 2.5f;
    
    opengv::points_t gt_points;
    opengv::bearingVectors_t compared_model_bearing_vectors;
    
    for (int p = 0; p < kPointCount; ++ p) {
      // Pick random pixel position
      Vec2f pixel_position = (0.5f * Vec2f::Random() + Vec2f(0.5f, 0.5f)).cwiseProduct(Vec2f(gt_model->width(), gt_model->height()));
      
      // Unproject the position with both models
      Vec3d gt_direction;
      Vec3d compared_direction;
      if (!gt_model->Unproject(pixel_position.x(), pixel_position.y(), &gt_direction) ||
          !compared_model->Unproject(pixel_position.x(), pixel_position.y(), &compared_direction)) {
        -- p;
        continue;
      }
      gt_direction.normalize();
      
      // Add a 3D point with the ground truth direction and random distance
      float distance = kMinDistance + ((rand() % 10000) / 10000.f) * (kMaxDistance - kMinDistance);
      gt_points.push_back(gt_direction * distance);
      
      // Store the compared direction
      compared_model_bearing_vectors.push_back(compared_direction.normalized());
    }
    
    // Optimize the camera pose with the ground truth 3D points and the compared model.
    opengv::absolute_pose::CentralAbsoluteAdapter adapter(
        compared_model_bearing_vectors,
        gt_points);
    adapter.sett(Vec3d::Zero());
    adapter.setR(Mat3d::Identity());
    opengv::transformation_t global_tr_image_matrix = opengv::absolute_pose::optimize_nonlinear(adapter);
    
    // Compute the camera center distance change from this process.
    Vec3d translation = global_tr_image_matrix.block<3, 1>(0, 3);
    float camera_error_distance = translation.norm();
    // LOG(INFO) << "[" << trial << "] error [mm]: " << (1000 * camera_error_distance);
    
    error_distance_mean.AddData(camera_error_distance);
    errors[trial] = camera_error_distance;
  }
  
  std::sort(errors.begin(), errors.end());
  double median_error = errors[errors.size() / 2];
  
  // Report the average and median camera center distance change.
  LOG(INFO) << "Average error [mm]: " << (1000 * error_distance_mean.ComputeArithmeticMean());
  LOG(INFO) << "Median error [mm]: " << (1000 * median_error);
  
  return EXIT_SUCCESS;
}

}
