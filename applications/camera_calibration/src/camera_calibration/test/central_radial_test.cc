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
#include "camera_calibration/calibration_report.h"
#include "camera_calibration/dataset.h"
#include "camera_calibration/fitting_report.h"
#include "camera_calibration/models/central_radial.h"
#include "camera_calibration/test/util.h"
#include "camera_calibration/util.h"

using namespace vis;

TEST(CentralRadial, OptimizeJointly) {
  constexpr int num_cameras = 1;
  constexpr bool use_cuda = false;
  
  // NOTE: If fixing something, make sure that any potential Gauge fixing residuals
  //       are disabled for DoFs that are not free anymore then!
  constexpr bool kDebugFixPoints = false;
  constexpr bool kDebugFixPoses = false;
  constexpr bool kDebugFixRigPoses = false;
  constexpr bool kDebugFixIntrinsics = false;
  
  constexpr bool kDebugPointProjections = false;
  
  constexpr int kCameraWidth = 600;
  constexpr int kCameraHeight = 400;
  
  srand(2);
  BAState state;
  
  for (int camera_index = 0; camera_index < num_cameras; ++ camera_index) {
    CentralRadialModel* model = new CentralRadialModel(
        kCameraWidth,
        kCameraHeight,
        8);
    state.intrinsics.emplace_back(shared_ptr<CameraModel>(model));
    state.camera_tr_rig.emplace_back((camera_index == 0) ? SE3d() : SE3d::exp(0.05 * Matrix<double, 6, 1>::Random()));
    
    Matrix<double, Eigen::Dynamic, 1>& params = model->parameters();
    params(0) = 320 + 20 * camera_index;  // fx
    params(1) = 300;  // fy
    params(2) = kCameraWidth / 2;  // cx
    params(3) = kCameraHeight / 2;  // cy
    params(4) = 0.001;  // p1
    params(5) = -0.001;  // p2
    params(6) = 0.002;  // sx1
    params(7) = -0.002;  // sy1
    params(8) = 0.01;  // radial0
    params(9) = 0.02;  // radial1
    params(10) = 0.03;  // radial2
    params(11) = 0.04;  // radial3
    params(12) = 0.05;  // radial4
    params(13) = 0.06;  // radial5
    params(14) = 0.07;  // radial6
    params(15) = 0.08;  // radial7
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
  
  // Disturb the cameras' parameters
  if (!kDebugFixIntrinsics) {
    for (int camera_index = 0; camera_index < num_cameras; ++ camera_index) {
      CentralRadialModel* model = dynamic_cast<CentralRadialModel*>(state.intrinsics[camera_index].get());
      ASSERT_TRUE(model);
      
      Matrix<double, Eigen::Dynamic, 1>& params = model->parameters();
      params(0) += 20 * ((rand() % 10000) / 5000.f - 1.f);
      params(1) += 20 * ((rand() % 10000) / 5000.f - 1.f);
      params(2) += 20 * ((rand() % 10000) / 5000.f - 1.f);
      params(3) += 20 * ((rand() % 10000) / 5000.f - 1.f);
      params(4) += 0.0005 * ((rand() % 10000) / 5000.f - 1.f);
      params(5) += 0.0005 * ((rand() % 10000) / 5000.f - 1.f);
      params(6) += 0.0005 * ((rand() % 10000) / 5000.f - 1.f);
      params(7) += 0.0005 * ((rand() % 10000) / 5000.f - 1.f);
      params(8) += 0.005 * ((rand() % 10000) / 5000.f - 1.f);
      params(9) += 0.005 * ((rand() % 10000) / 5000.f - 1.f);
      params(10) += 0.005 * ((rand() % 10000) / 5000.f - 1.f);
      params(11) += 0.005 * ((rand() % 10000) / 5000.f - 1.f);
      params(12) += 0.005 * ((rand() % 10000) / 5000.f - 1.f);
      params(13) += 0.005 * ((rand() % 10000) / 5000.f - 1.f);
      params(14) += 0.005 * ((rand() % 10000) / 5000.f - 1.f);
      params(15) += 0.005 * ((rand() % 10000) / 5000.f - 1.f);
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
  for (int i = 0; i < 80; ++ i) {
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
          SchurMode::Dense,
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
    
    
    // Debug visualization
    constexpr bool kShowDebugVisualization = false;
    if (kShowDebugVisualization) {
      for (int camera_index = 0; camera_index < num_cameras; ++ camera_index) {
        Image<Vec3u8> visualization;
        CreateReprojectionErrorMagnitudeVisualization(
            dataset,
            camera_index,
            state,
            5.0f,
            &visualization);
        
        static vector<ImageDisplay> reprojection_error_display;
        reprojection_error_display.resize(num_cameras);
        reprojection_error_display[camera_index].Clear();
        
        for (int imageset_index = 0; imageset_index < dataset.ImagesetCount(); ++ imageset_index) {
          if (!state.image_used[imageset_index]) {
            continue;
          }
          
          const SE3d& image_tr_global = state.image_tr_global(camera_index, imageset_index);
          Mat3d image_r_global = image_tr_global.rotationMatrix();
          const Vec3d& image_t_global = image_tr_global.translation();
          
          shared_ptr<const Imageset> imageset = dataset.GetImageset(imageset_index);
          const vector<PointFeature>& features = imageset->FeaturesOfCamera(camera_index);
          
          for (const PointFeature& feature : features) {
            Vec3d local_point = image_r_global * state.points[feature.index] + image_t_global;
            Vec2d pixel;
            if (state.intrinsics[camera_index]->Project(local_point, &pixel)) {
              Vec2d reprojection_error = pixel - feature.xy.cast<double>();
              
              Vec3u8 color;
              if (reprojection_error.norm() > 50) {
                color = Vec3u8(255, 0, 0);
              } else if (reprojection_error.norm() > 10) {
                color = Vec3u8(255, 100, 100);
              } else {
                color = Vec3u8(200, 200, 200);
              }
              reprojection_error_display[camera_index].AddSubpixelLinePixelCornerConv(
                  pixel, feature.xy, color);
            }
          }
        }
        ostringstream camera_name;
        camera_name << "camera " << camera_index;
        reprojection_error_display[camera_index].Update(visualization, "Reprojection errors " + camera_name.str());
      }
      std::getchar();
    }
  }
  
  const double result_threshold = use_cuda ? 8.f : 1e-3f;  // TODO: Can we get the CUDA version to be more accurate?
  EXPECT_LE(last_cost, num_cameras * result_threshold);
}

TEST(CentralRadial, ProjectionAndUnprojection) {
  constexpr int kCameraWidth = 640 / 4;
  constexpr int kCameraHeight = 480 / 4;
  
  CentralRadialModel model(kCameraWidth, kCameraHeight, 8);
  model.parameters() <<
      kCameraHeight / 1.5,
      kCameraHeight / 1.5,
      kCameraWidth / 2,
      kCameraHeight / 2,
      0, 0, 0, 0,
      0.01, 0.02, 0.03, 0.04,
      0.05, 0.06, 0.07, 0.08;
  
  TestProjectUnproject(model);
}

TEST(CentralRadial, FitToDenseModel) {
  constexpr int kCameraWidth = 640 / 4;
  constexpr int kCameraHeight = 480 / 4;
  
  // Define the ground truth model
  CentralRadialModel model(kCameraWidth, kCameraHeight, 8);
  model.parameters() <<
      kCameraHeight / 1.5,
      kCameraHeight / 1.5,
      kCameraWidth / 2 + 1,
      kCameraHeight / 2 + 2,
      0.001, -0.0001, 0.0001, 0.0002,
      0.01, 0.02, 0.03, 0.04,
      0.05, 0.06, 0.07, 0.08;
  
  Image<Vec3d> dense_model(model.width(), model.height());
  for (int y = 0; y < model.height(); ++ y) {
    for (int x = 0; x < model.width(); ++ x) {
      Vec3d direction;
      if (!model.Unproject(x + 0.5f, y + 0.5f, &direction)) {
        LOG(FATAL) << "Error in test setup (or CentralRadial::Unproject()): ground truth model values produce unprojection failure";
      }
      dense_model(x, y) = direction;
      if (direction.hasNaN()) {
        LOG(FATAL) << "Error in test setup (or CentralRadial::Unproject()): ground truth model values produce NaN direction when unprojecting from image area";
      }
    }
  }
  
  // Try to fit the model
  CentralRadialModel estimated_model(dense_model.width(), dense_model.height(), 8);
  estimated_model.FitToDenseModel(dense_model, /*subsample_step*/ 1);
  
  // Verify that the model was fitted well
  constexpr double kEpsilon = 1e-3f;
  int estimated_model_unprojection_failures = 0;
  int num_failures = 0;
  for (int y = 0; y < model.height(); ++ y) {
    for (int x = 0; x < model.width(); ++ x) {
      Vec3d gt_direction;
      if (!model.Unproject(x + 0.5f, y + 0.5f, &gt_direction)) {
        LOG(FATAL) << "Error in test setup (or CentralRadial::Unproject()): ground truth model values produce unprojection failure";
      }
      
      Vec3d estimated_direction;
      if (!estimated_model.Unproject(x + 0.5f, y + 0.5f, &estimated_direction)) {
        ++ estimated_model_unprojection_failures;
        continue;
      }
      
      if ((gt_direction - estimated_direction).norm() > kEpsilon) {
        ++ num_failures;
      }
    }
  }
  EXPECT_EQ(0, estimated_model_unprojection_failures);
  EXPECT_EQ(0, num_failures);
}
