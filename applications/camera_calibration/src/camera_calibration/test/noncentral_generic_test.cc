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


#include <libvis/logging.h>
#include <gtest/gtest.h>

#include "libvis/camera.h"
#include "libvis/eigen.h"
#include "libvis/image.h"
#include "libvis/image_display.h"
#include "libvis/lm_optimizer.h"
#include "libvis/sophus.h"

#include "camera_calibration/bundle_adjustment/ba_state.h"
#include "camera_calibration/bundle_adjustment/joint_optimization.h"
#include "camera_calibration/calibration.h"
#include "camera_calibration/dataset.h"
#include "camera_calibration/models/noncentral_generic.h"
#include "camera_calibration/local_parametrizations/line_parametrization.h"

using namespace vis;

TEST(NoncentralGenericBSpline, OrthogonalCameraProjectionAndUnprojection) {
  // Allocate camera
  constexpr int kCameraWidth = 100;
  constexpr int kCameraHeight = 100;
  NoncentralGenericModel model(
      /*grid_resolution_x*/ 4,
      /*grid_resolution_y*/ 4,
      0, 0,
      kCameraWidth - 1,
      kCameraHeight - 1,
      kCameraWidth,
      kCameraHeight);
  
  // Define camera model
  Image<Vec3d> point_grid(4, 4);
  Image<Vec3d> direction_grid(4, 4);
  for (int y = 0; y < 4; ++ y) {
    for (int x = 0; x < 4; ++ x) {
      point_grid(x, y) = Vec3d(x, y, 0);
      direction_grid(x, y) = Vec3d(0, 0, 1);
    }
  }
  model.SetPointGrid(point_grid);
  model.SetDirectionGrid(direction_grid);
  
  // Tests
  Line3d line;
  bool ok;
  EXPECT_TRUE(ok = model.Unproject(0.5 * kCameraWidth, 0.5 * kCameraHeight, &line));
  if (ok) {
    EXPECT_NEAR(line.origin().x(), 1.5f, 1e-5f);
    EXPECT_NEAR(line.origin().y(), 1.5f, 1e-5f);
    EXPECT_NEAR(line.direction().x(), 0, 1e-5f);
    EXPECT_NEAR(line.direction().y(), 0, 1e-5f);
    EXPECT_NEAR(fabs(line.direction().z()), 1, 1e-5f);
  }
  
  Vec2d pixel;
  EXPECT_TRUE(ok = model.Project(Vec3d(1.5, 1.5, 42.12345), &pixel));
  if (ok) {
    EXPECT_NEAR(pixel.x(), 0.5 * kCameraWidth, 1e-5f);
    EXPECT_NEAR(pixel.y(), 0.5 * kCameraHeight, 1e-5f);
  }
  
  double test_grid_x = 1.1;
  double test_grid_y = 1.2;
  EXPECT_TRUE(ok = model.Project(Vec3d(test_grid_x, test_grid_y, 42.12345), &pixel));
  if (ok) {
    EXPECT_NEAR(pixel.x(), model.GridPointToPixelCornerConv(test_grid_x, test_grid_y).x(), 1e-5f);
    EXPECT_NEAR(pixel.y(), model.GridPointToPixelCornerConv(test_grid_x, test_grid_y).y(), 1e-5f);
  }
  
  // Test a point very close to the camera image border
  test_grid_x = 1.001;
  test_grid_y = 1.999;
  EXPECT_TRUE(ok = model.Project(Vec3d(test_grid_x, test_grid_y, 42.12345), &pixel));
  if (ok) {
    EXPECT_NEAR(pixel.x(), model.GridPointToPixelCornerConv(test_grid_x, test_grid_y).x(), 1e-5f);
    EXPECT_NEAR(pixel.y(), model.GridPointToPixelCornerConv(test_grid_x, test_grid_y).y(), 1e-5f);
  }
}

TEST(NoncentralGenericBSpline, OptimizeJointly) {
  BAState state;
  
  constexpr int kCameraWidth = 640;
  constexpr int kCameraHeight = 480;
  
  NoncentralGenericModel* model = new NoncentralGenericModel(
      /*grid_resolution_x*/ 8,
      /*grid_resolution_y*/ 6,
      0, 0,
      kCameraWidth - 1,
      kCameraHeight - 1,
      kCameraWidth,
      kCameraHeight);
  constexpr int camera_index = 0;
  state.intrinsics = {shared_ptr<CameraModel>(model)};
  state.camera_tr_rig.resize(1, SE3d());
  
  // Define camera model somewhat arbitrarily
  Image<Vec3d> point_grid(4, 4);
  Image<Vec3d> direction_grid(4, 4);
  for (int y = 0; y < 4; ++ y) {
    for (int x = 0; x < 4; ++ x) {
      point_grid(x, y) = Vec3d(-1.5f + x, -1.5f + y, 0);
      direction_grid(x, y) = Vec3d(0, 0.05 * x, 1).normalized();
    }
  }
  model->SetPointGrid(point_grid);
  model->SetDirectionGrid(direction_grid);
  
  // Define the ground truth structure.
  constexpr int kNumPoints = 50;
  vector<Vec3d> gt_points(kNumPoints);
  for (int i = 0; i < kNumPoints; ++ i) {
    gt_points[i] = 0.3f * Vec3d::Random();
  }
  
  // Define the ground truth poses, and the matches.
  constexpr int kNumPoses = 20;
  vector<SE3d> gt_image_tr_global(kNumPoses);
  
  Dataset dataset(state.intrinsics.size());
  dataset.SetImageSize(camera_index, Vec2i(kCameraWidth, kCameraHeight));
  
  for (int i = 0; i < kNumPoses; ++ i) {
    gt_image_tr_global[i] = SE3d::exp(0.05 * Matrix<double, 6, 1>::Random()) * SE3d(Mat3d::Identity(), Vec3d(0, 0, 1));
    
    shared_ptr<Imageset> new_imageset = dataset.NewImageset();
    vector<PointFeature>& features = new_imageset->FeaturesOfCamera(0);
    
    for (int p = 0; p < kNumPoints; ++ p) {
      Vec2d projection;
      if (model->Project(
              gt_image_tr_global[i] * gt_points[p],  // slower than necessary!
              &projection)) {
        features.emplace_back(projection.cast<float>(), p);
        features.back().index = p;  // replaces state.feature_id_to_points_index and ComputeFeatureIdToPointsIndex()
      }
    }
//     LOG(INFO) << "#features in image " << i << ": " << features.size() << " / " << kNumPoints;
  }
  
  // Disturb the poses and points (except the first pose)
  state.points = gt_points;
  for (int i = 0; i < kNumPoints; ++ i) {
    state.points[i] += 0.05f * Vec3d::Random();
  }
  
  state.rig_tr_global = gt_image_tr_global;
  for (int i = 1; i < kNumPoses; ++ i) {  // skip first pose
    state.rig_tr_global[i] *= SE3d::exp(0.04 * Matrix<double, 6, 1>::Random());
  }
  
//   // Disturb the grids
//   Image<Vec3d> estimated_point_grid = point_grid;
//   Image<Vec3d> estimated_direction_grid = direction_grid;
//   for (int y = 0; y < 4; ++ y) {
//     for (int x = 0; x < 4; ++ x) {
//       estimated_point_grid(x, y) += 0.05 * Vec3d::Random();
//       estimated_direction_grid(x, y) = (direction_grid(x, y) + 0.06 * Vec3d::Random()).normalized();
//     }
//   }
//   model->SetPointGrid(estimated_point_grid);
//   model->SetDirectionGrid(estimated_direction_grid);
  
  // Optimize poses and points.
  state.image_used.resize(kNumPoses, true);
  double final_cost = OptimizeJointly(
      dataset,
      &state,
      /*max_iteration_count*/ 50,
      /*init_lambda*/ -1,
      /*numerical_diff_delta*/ 0.001,
      /*regularization_weight*/ 0,
      /*localize_only*/ false,
      /*eliminate_points*/ false,
      SchurMode::Dense,
      /*final_lambda*/ nullptr,
      /*performed_an_iteration*/ nullptr,
      /*debug_verify_cost*/ true,
      /*debug_fix_points*/ false,
      /*debug_fix_poses*/ false,
      /*debug_fix_rig_poses*/ false,
      /*debug_fix_intrinsics*/ false,
      /*print_progress*/ true);
  
//   // Scale the result to match the ground truth scale.
//   double factor =
//       (gt_points[1] - gt_points[0]).norm() /
//       (estimated_points[1] - estimated_points[0]).norm();
//   
//   Vec3d scaling_origin = estimated_image_tr_world[0].inverse().translation();
//   
//   for (int i = 0; i < kNumPoints; ++ i) {
//     estimated_points[i] =
//         scaling_origin + factor * (estimated_points[i] - scaling_origin);
//   }
//   
//   for (int i = 1; i < kNumPoses; ++ i) {
//     SE3d world_tr_image = estimated_image_tr_world[i].inverse();
//     world_tr_image.translation() =
//         scaling_origin + factor * (world_tr_image.translation() - scaling_origin);
//     estimated_image_tr_world[i] = world_tr_image.inverse();
//   }
//   
  // Compare the result to the ground truth.
  // TODO: This works for optimizing poses and geometry only.
  //       If jointly optimizing intrinsics, the intrinsics and the camera pose
  //       can rotate in opposite ways while leaving the resulting calibration
  //       constant, which means that the resulting poses might not exactly
  //       correspond to the ground truth, despite a correct result. Therefore,
  //       we currently only check that we get a near-zero final cost, which is
  //       sufficient to check that we get a good result, under the assumption
  //       that the cost computation is bug-free.
//   constexpr double kEpsilon = 1e-4f;
//   
//   for (int i = 0; i < kNumPoints; ++ i) {
//     EXPECT_LE((estimated_points[i] - gt_points[i]).norm(), kEpsilon) << "Wrong point: " << i;
//   }
//   
//   for (int i = 0; i < kNumPoses; ++ i) {
//     EXPECT_LE((estimated_image_tr_world[i].inverse() * gt_image_tr_global[i]).log().norm(), kEpsilon) << "Wrong pose: " << i;
//   }
  
  EXPECT_LE(final_cost, 2e-4f);
}
