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

#include <gtest/gtest.h>
#include <libvis/image.h>
#include <libvis/libvis.h>
#include <libvis/logging.h>

#include "camera_calibration/bundle_adjustment/ba_state.h"
#include "camera_calibration/bundle_adjustment/cuda_joint_optimization.h"
#include "camera_calibration/bundle_adjustment/joint_optimization.h"
#include "camera_calibration/dataset.h"

namespace vis {

template <typename ModelT, typename T>
void VerifyUnprojections(
    const ModelT& model,
    const T& dense_model) {
  constexpr double kEpsilon = 5e-4f;
  
  bool had_error = false;
  double max_eps = 0;
  // Image<double> cost_visualization(model.width(), model.height());
  for (u32 y = 0; y < model.height(); ++ y) {
    for (u32 x = 0; x < model.width(); ++ x) {
      Vec3d unprojection;
      if (!model.Unproject(x + 0.5f, y + 0.5f, &unprojection)) {
        continue;
      }
      Vec3d diff = unprojection - dense_model(x, y);
      double residual_cost = 0.5f * diff.squaredNorm();
      max_eps = std::max(max_eps, residual_cost);
      // cost_visualization(x, y) = residual_cost;
      if (!had_error) {
        EXPECT_LT(residual_cost, kEpsilon);
        if (residual_cost > kEpsilon) {
          LOG(INFO) << "Not reporting potential further issues in the same test to avoid log spam.";
          had_error = true;
        }
      }
    }
  }
  LOG(INFO) << "max_eps: " << max_eps;
  // ImageDisplay cost_visualization_display;
  // cost_visualization_display.Update(cost_visualization, "Residuals after FitToDenseModel()", 0, max_eps);
}


template <typename ModelT>
void TestProjectUnproject(const ModelT& camera) {
  int num_failures = 0;
  int num_unprojection_fails = 0;
  for (int y = 0; y < camera.height(); ++ y) {
    for (int x = 0; x < camera.width(); ++ x) {  
      Vec3d direction;
      if (!camera.Unproject(x + 0.5f, y + 0.5f, &direction)) {
        ++ num_unprojection_fails;
        continue;
      }
      Vec2d pixel;
      ASSERT_TRUE(camera.Project(direction, &pixel));
      
      constexpr double kEpsilon = 1e-3f;
      if (fabs(x + 0.5f - pixel.x()) > kEpsilon ||
          fabs(y + 0.5f - pixel.y()) > kEpsilon) {
        ++ num_failures;
      }
      
      // ASSERT_NEAR(x + 0.5f, pixel.x(), kEpsilon) << "x: " << x << ", y: " << y << ", pixel.x(): " << pixel.x() << ", pixel.y(): " << pixel.y();
      // ASSERT_NEAR(y + 0.5f, pixel.y(), kEpsilon) << "x: " << x << ", y: " << y << ", pixel.x(): " << pixel.x() << ", pixel.y(): " << pixel.y();
    }
  }
  
  if (num_unprojection_fails > 0) {
    LOG(WARNING) << "There were unprojection failures: " << num_unprojection_fails;
  }
  EXPECT_EQ(0, num_failures) << "Total test count is: " << camera.width() * camera.height();
}


// Main requirements on ModelT:
// - Must have a grid
// - Must unproject to a direction vector
template <typename ModelT>
void TestProjectUnproject() {
  srand(0);
  
  constexpr int kCameraWidth = 640;
  constexpr int kCameraHeight = 480;
  
  ModelT model(
      /*grid_resolution_x*/ 8,
      /*grid_resolution_y*/ 6,
      10, 20,
      kCameraWidth - 5,
      kCameraHeight - 8,
      kCameraWidth,
      kCameraHeight);
  
  Image<Vec3d> grid(model.grid().size());
  for (u32 y = 0; y < grid.height(); ++ y) {
    for (u32 x = 0; x < grid.width(); ++ x) {
      grid(x, y) = Vec3d(x, y, 1).normalized();
    }
  }
  model.SetGrid(grid);
  
  for (int i = 0; i < 400; ++ i) {
    Vec2d pixel = Vec2d(10, 20) + Vec2d::Random().cwiseAbs().cwiseProduct(Vec2d(kCameraWidth - 14, kCameraHeight - 27));
    
    Vec3d direction1;
    bool ok1 = model.Unproject(pixel.x(), pixel.y(), &direction1);
    EXPECT_TRUE(ok1);
    
    Vec3d direction2;
    Matrix<double, 3, 2> direction_wrt_pixel;
    bool ok2 = model.UnprojectWithJacobian(pixel.x(), pixel.y(), &direction2, &direction_wrt_pixel);
    EXPECT_TRUE(ok2);
    
    if (ok1 && ok2) {
      EXPECT_NEAR(direction1.x(), direction2.x(), 1e-5);
      EXPECT_NEAR(direction1.y(), direction2.y(), 1e-5);
      EXPECT_NEAR(direction1.z(), direction2.z(), 1e-5);
    }
    
    if (ok1) {
      Vec2d reprojection;
      bool ok3 = model.ProjectDirection(direction1, &reprojection);
      EXPECT_TRUE(ok3) << "Failed to re-project the un-projection of pixel (" << pixel.x() << ", " << pixel.y() << ")";
      if (ok3) {
        EXPECT_NEAR(pixel.x(), reprojection.x(), 1e-4);
        EXPECT_NEAR(pixel.y(), reprojection.y(), 1e-4);
      }
    }
  }
}


template <typename ModelT>
void TestFitCentralModel(ModelT* estimated_model, const Image<Vec3d>& dense_model, Mat3d ground_truth_parametric_r_dense, const ModelT& ground_truth_model) {
  // Try to fit the model
  Mat3d parametric_r_dense_estimate = Mat3d::Identity();
  estimated_model->FitToDenseModel(dense_model, &parametric_r_dense_estimate, /*subsample_step*/ 1);
  
  // Verify that the model was fitted well
  // NOTE: Comparing the parameters directly will not work properly if different
  //       combinations of parameters can result in very similar models
  int num_failures = 0;
  int num_unprojection_fails = 0;
  for (int y = 0; y < ground_truth_model.height(); ++ y) {
    for (int x = 0; x < ground_truth_model.width(); ++ x) {
      Vec3d direction;
      CHECK(ground_truth_model.Unproject(x + 0.5f, y + 0.5f, &direction));
      
      Vec3d estimated_direction;
      if (!estimated_model->Unproject(x + 0.5f, y + 0.5f, &estimated_direction)) {
        ++ num_unprojection_fails;
        continue;
      }
      
      constexpr double kEpsilon = 1e-4f * 1e-4f;
      if ((direction - estimated_direction).squaredNorm() > kEpsilon) {
        ++ num_failures;
      }
    }
  }
  
  if (num_unprojection_fails > 0) {
    LOG(WARNING) << "There were unprojection failures: " << num_unprojection_fails;
  }
  EXPECT_EQ(0, num_failures) << "Total test count is: " << ground_truth_model.width() * ground_truth_model.height();
  EXPECT_NEAR((parametric_r_dense_estimate - ground_truth_parametric_r_dense).norm(), 0, 0.001f);
  
  // Image<Vec3u8> estimated_model_visualization;
  // VisualizeModelDirections(estimated_model, &estimated_model_visualization);
  // ImageDisplay estimated_model_visualization_display;
  // estimated_model_visualization_display.Update(estimated_model_visualization, "Estimated model");
  // std::getchar();
}


// Main requirements on ModelT:
// - Must support FitToDenseModel()
// - Must support FitToPixelDirections()
template <typename ModelT>
void TestModelOptimization() {
  constexpr int kCameraWidth = 640;
  constexpr int kCameraHeight = 480;
  
  ModelT model(
      /*grid_resolution_x*/ 8,
      /*grid_resolution_y*/ 6,
      0, 0,
      kCameraWidth - 1,
      kCameraHeight - 1,
      kCameraWidth,
      kCameraHeight);
  
  // Use a pinhole camera model to define the ground truth.
  // fx, fy, cx, cy
  float pinhole_parameters[] = {kCameraHeight / 2, kCameraHeight / 2, kCameraWidth / 2, kCameraHeight / 2};
  PinholeCamera4f ground_truth(kCameraWidth, kCameraHeight, pinhole_parameters);
  
  // Test: FitToDenseModel()
  // This initializes the intrinsics in the model.
  Image<Vec3d> dense_model(kCameraWidth, kCameraHeight);
  for (u32 y = 0; y < kCameraHeight; ++ y) {
    for (u32 x = 0; x < kCameraWidth; ++ x) {
      dense_model(x, y) = ground_truth.UnprojectFromPixelCenterConv(Vec2i(x, y)).cast<double>().normalized();
    }
  }
  
  model.FitToDenseModel(dense_model, 2);
  
  VerifyUnprojections(model, dense_model);
  
  // Test: FitToPixelDirections()
  // The optimization here is supposed to change the intrinsics (that were
  // initialized with FitToDenseModel() above) to the shifted state.
  {
    vector<Vec2d> pixels;
    vector<Vec3d> directions;
    
    float shifted_pinhole_parameters[] = {kCameraHeight / 2, kCameraHeight / 2, kCameraWidth / 2 - 10, kCameraHeight / 2 + 20};
    PinholeCamera4f shifted_ground_truth(kCameraWidth, kCameraHeight, shifted_pinhole_parameters);
    
    constexpr int kStep = 10;
    for (u32 y = 0; y < kCameraHeight; y += kStep) {
      for (u32 x = 0; x < kCameraWidth; x += kStep) {
        pixels.push_back(Vec2d(x + 0.5f, y + 0.5f));
        directions.push_back(shifted_ground_truth.UnprojectFromPixelCenterConv(Vec2i(x, y)).cast<double>().normalized());
      }
    }
    
    model.FitToPixelDirections(pixels, directions, /*max_iteration_count*/ 10);
    
    VerifyUnprojections(model, [&](int x, int y) -> Vec3d {
      return shifted_ground_truth.UnprojectFromPixelCenterConv(Vec2i(x, y)).cast<double>().normalized();
    });
  }
}


// Main requirements on ModelT:
// - Must have a grid
// - Must support FitToDenseModel()
template <typename ModelT>
void TestOptimizeJointly(bool use_cuda, SchurMode schur_mode, int num_cameras) {
  // NOTE: If fixing something, make sure that any potential Gauge fixing residuals
  //       are disabled for DoFs that are not free anymore then!
  constexpr bool kDebugFixPoints = false;
  constexpr bool kDebugFixPoses = false;
  constexpr bool kDebugFixRigPoses = false;
  constexpr bool kDebugFixIntrinsics = false;
  
  constexpr bool kDebugPointProjections = false;
  
  constexpr int kCameraWidth = 600;
  constexpr int kCameraHeight = 400;
  
  srand(0);
  BAState state;
  
  for (int camera_index = 0; camera_index < num_cameras; ++ camera_index) {
    ModelT* model = new ModelT(
        /*grid_resolution_x*/ 5,
        /*grid_resolution_y*/ 5,
        0, 0,
        kCameraWidth - 1,
        kCameraHeight - 1,
        kCameraWidth,
        kCameraHeight);
    state.intrinsics.emplace_back(shared_ptr<CameraModel>(model));
    state.camera_tr_rig.emplace_back((camera_index == 0) ? SE3d() : SE3d::exp(0.05 * Matrix<double, 6, 1>::Random()));
    
    // Use a pinhole camera model to define the camera.
    // fx, fy, cx, cy
    float pinhole_parameters[] = {kCameraHeight / 2.f + 2.f * camera_index, kCameraHeight / 2, kCameraWidth / 2, kCameraHeight / 2};
    PinholeCamera4f ground_truth(kCameraWidth, kCameraHeight, pinhole_parameters);
    
    Image<Vec3d> dense_model(kCameraWidth, kCameraHeight);
    for (u32 y = 0; y < kCameraHeight; ++ y) {
      for (u32 x = 0; x < kCameraWidth; ++ x) {
        dense_model(x, y) = ground_truth.UnprojectFromPixelCenterConv(Vec2i(x, y)).cast<double>().normalized();
      }
    }
    
    model->FitToDenseModel(dense_model, 2);
  }
  
  // Define the ground truth structure.
  constexpr int kNumPoints = 150;
  vector<Vec3d> gt_points(kNumPoints);
  for (int i = 0; i < kNumPoints; ++ i) {
    gt_points[i] = Vec3d::Random().cwiseProduct(Vec3d(6.5, 3.5, 1));
  }
  state.points = gt_points;
  
  // Define the ground truth poses, and the matches.
  constexpr int kNumPoses = 100;
  vector<SE3d> gt_rig_tr_global(kNumPoses);
  
  Dataset dataset(state.intrinsics.size());
  for (int camera_index = 0; camera_index < num_cameras; ++ camera_index) {
    dataset.SetImageSize(camera_index, Vec2i(state.intrinsics[camera_index]->width(), state.intrinsics[camera_index]->height()));
  }
  
  ImageDisplay debug_projections;
  for (int i = 0; i < kNumPoses; ++ i) {
    gt_rig_tr_global[i] = SE3d::exp(0.05 * Matrix<double, 6, 1>::Random()) * SE3d(Mat3d::Identity(), Vec3d(0, 0, 5) + Vec3d::Random());
    
    shared_ptr<Imageset> new_imageset = dataset.NewImageset();
    for (int camera_index = 0; camera_index < num_cameras; ++ camera_index) {
      SE3d camera_tr_global = state.camera_tr_rig[camera_index] * gt_rig_tr_global[i];
      
      vector<PointFeature>& features = new_imageset->FeaturesOfCamera(camera_index);
      for (int p = 0; p < kNumPoints; ++ p) {
        Vec2d projection;
        if (state.intrinsics[camera_index]->Project(
                camera_tr_global * gt_points[p],  // slower than necessary!
                &projection)) {
          features.emplace_back(projection.cast<float>(), p);
          features.back().index = p;  // replaces state.feature_id_to_points_index and ComputeFeatureIdToPointsIndex()
          if (kDebugPointProjections) {
            debug_projections.AddSubpixelDotPixelCornerConv(features.back().xy, Vec3u8(255, 255, 255));
          }
        }
      }
      if (kDebugPointProjections) {
        LOG(INFO) << "#features in image " << i << ": " << features.size() << " / " << kNumPoints;
      }
    }
  }
  state.rig_tr_global = gt_rig_tr_global;
  
  Image<u8> dummy;
  if (kDebugPointProjections) {
    dummy.SetSize(kCameraWidth, kCameraHeight);
    dummy.SetTo(static_cast<u8>(0));
    debug_projections.Update(dummy, "Debug projections");
  }
  
  // Disturb the points
  if (!kDebugFixPoints) {
    for (int i = 0; i < kNumPoints; ++ i) {
      state.points[i] += 0.05f * Vec3d::Random();
    }
  }
  
  // Disturb the poses, except the first one
  if (!kDebugFixPoses) {
    for (int i = 0; i < kNumPoses; ++ i) {
      state.rig_tr_global[i] *= SE3d::exp(0.04 * Matrix<double, 6, 1>::Random());
    }
  }
  
  // For camera rigs, disturb the camera_tr_rig poses
  if (!kDebugFixRigPoses) {
    for (int i = 0; i < state.intrinsics.size(); ++ i) {
      state.camera_tr_rig[i] *= SE3d::exp(0.04 * Matrix<double, 6, 1>::Random());
    }
  }
  
  // Disturb the cameras' grids
  if (!kDebugFixIntrinsics) {
    for (int camera_index = 0; camera_index < num_cameras; ++ camera_index) {
      ModelT* model = dynamic_cast<ModelT*>(state.intrinsics[camera_index].get());
      ASSERT_TRUE(model);
      
      Image<Vec3d> estimated_grid = model->grid();
      for (int y = 0; y < estimated_grid.height(); ++ y) {
        for (int x = 0; x < estimated_grid.width(); ++ x) {
          estimated_grid(x, y) = (estimated_grid(x, y) + 0.02 * Vec3d::Random()).normalized();
        }
      }
      model->SetGrid(estimated_grid);
    }
  }
  
  if (kDebugPointProjections) {
    const int camera_index = 0;
    
    for (int i = 0; i < kNumPoses; ++ i) {
      for (int p = 0; p < kNumPoints; ++ p) {
        Vec2d projection;
        if (state.intrinsics[camera_index]->Project(
                gt_rig_tr_global[i] * gt_points[p],  // slower than necessary!
                &projection)) {
          debug_projections.AddSubpixelDotPixelCornerConv(projection, Vec3u8(255, 0, 0));
        }
      }
    }
    dummy.SetTo(static_cast<u8>(80));
    ostringstream camera_index_str;
    camera_index_str << camera_index;
    debug_projections.Update(dummy, "Debug projections (camera " + camera_index_str.str() + ")");
  }
  
  // Optimize poses and points.
  state.image_used.resize(kNumPoses, true);
  
  double last_cost = numeric_limits<double>::infinity();
  double lambda = -1;
  for (int i = 0; i < 20 * num_cameras; ++ i) {
    if (use_cuda) {
      OptimizationReport report = CudaOptimizeJointly(
          dataset,
          &state,
          /*max_iteration_count*/ 1,
          /*int max_inner_iterations*/ 50,  // TODO: tune
          lambda,
          /*numerical_diff_delta*/ 0.0001,  // TODO: test changing this
          /*regularization_weight*/ 0,
          &lambda,
          /*debug_verify_cost*/ false,
          kDebugFixPoints,
          kDebugFixPoses,
          kDebugFixRigPoses,
          kDebugFixIntrinsics,
          /*print_progress*/ true);
      last_cost = report.final_cost;
    } else {
      bool performed_an_iteration;
      last_cost = OptimizeJointly(
          dataset,
          &state,
          /*max_iteration_count*/ 1,  // TODO: 20
          lambda,
          /*numerical_diff_delta*/ 0.0001,  // TODO: test changing this
          /*regularization_weight*/ 0,
          /*localize_only*/ false,
          /*eliminate_points*/ false,
          schur_mode,
          &lambda,
          &performed_an_iteration,
          /*debug_verify_cost*/ i == 0,
          kDebugFixPoints,
          kDebugFixPoses,
          kDebugFixRigPoses,
          kDebugFixIntrinsics,
          /*print_progress*/ true);
      if (!performed_an_iteration) {
        break;
      }
    }
    
    
//     // DEBUG VISUALIZATION
//     for (int camera_index = 0; camera_index < num_cameras; ++ camera_index) {
//       Image<Vec3u8> visualization;
//       CreateReprojectionErrorMagnitudeVisualization(
//           dataset,
//           camera_index,
//           state,
//           5.0f,
//           &visualization);
//       
//       static vector<ImageDisplay> reprojection_error_display;
//       reprojection_error_display.resize(num_cameras);
//       reprojection_error_display[camera_index].Clear();
//       
//       for (int imageset_index = 0; imageset_index < dataset.ImagesetCount(); ++ imageset_index) {
//         if (!state.image_used[imageset_index]) {
//           continue;
//         }
//         
//         const SE3d& image_tr_global = state.image_tr_global(camera_index, imageset_index);
//         Mat3d image_r_global = image_tr_global.rotationMatrix();
//         const Vec3d& image_t_global = image_tr_global.translation();
//         
//         shared_ptr<const Imageset> imageset = dataset.GetImageset(imageset_index);
//         const vector<PointFeature>& features = imageset->FeaturesOfCamera(camera_index);
//         
//         for (const PointFeature& feature : features) {
//           Vec3d local_point = image_r_global * state.points[feature.index] + image_t_global;
//           Vec2d pixel;
//           if (state.intrinsics[camera_index]->Project(local_point, &pixel)) {
//             Vec2d reprojection_error = pixel - feature.xy.cast<double>();
//             
//             Vec3u8 color;
//             if (reprojection_error.norm() > 50) {
//               color = Vec3u8(255, 0, 0);
//             } else if (reprojection_error.norm() > 10) {
//               color = Vec3u8(255, 100, 100);
//             } else {
//               color = Vec3u8(200, 200, 200);
//             }
//             reprojection_error_display[camera_index].AddSubpixelLinePixelCornerConv(
//                 pixel, feature.xy, color);
//           }
//         }
//       }
//       ostringstream camera_name;
//       camera_name << "camera " << camera_index;
//       reprojection_error_display[camera_index].Update(visualization, "Reprojection errors " + camera_name.str());
//     }
//     std::getchar();
//     // END DEBUG VISUALIZATION
  }
  
//   // Scale the result to match the ground truth scale.
//   double factor =
//       (gt_points[1] - gt_points[0]).norm() /
//       (state.points[1] - state.points[0]).norm();
//   
//   Vec3d scaling_origin = state.image_tr_global(camera_index, 0).inverse().translation();
//   
//   for (int i = 0; i < kNumPoints; ++ i) {
//     state.points[i] =
//         scaling_origin + factor * (state.points[i] - scaling_origin);
//   }
//   
//   for (int i = 1; i < kNumPoses; ++ i) {
//     SE3d world_tr_image = state.image_tr_global(camera_index, i).inverse();
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
//     EXPECT_LE((estimated_image_tr_world[i].inverse() * gt_rig_tr_global[i]).log().norm(), kEpsilon) << "Wrong pose: " << i;
//   }
  
  const double result_threshold = use_cuda ? 8.f : 1e-6f;  // TODO: Can we get the CUDA version to be more accurate?
  EXPECT_LE(last_cost, num_cameras * result_threshold);
  
//   std::getchar();
}

}
