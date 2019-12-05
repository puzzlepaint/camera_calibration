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

#include "camera_calibration/calibration_report.h"
#include "camera_calibration/models/central_thin_prism_fisheye.h"
#include "camera_calibration/test/util.h"

using namespace vis;

namespace {
template <typename ModelT>
void TestOptimizeJointlyParametric(bool use_cuda, int num_cameras, bool use_equidistant_projection) {
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
    ModelT* model = new ModelT(
        kCameraWidth,
        kCameraHeight,
        use_equidistant_projection);
    state.intrinsics.emplace_back(shared_ptr<CameraModel>(model));
    state.camera_tr_rig.emplace_back((camera_index == 0) ? SE3d() : SE3d::exp(0.05 * Matrix<double, 6, 1>::Random()));
    
    Matrix<double, Eigen::Dynamic, 1>& params = model->parameters();
    params(0) = 320 + 20 * camera_index;  // fx
    params(1) = 300;  // fy
    params(2) = kCameraWidth / 2;  // cx
    params(3) = kCameraHeight / 2;  // cy
    params(4) = 0.04;  // k1
    params(5) = -0.03;  // k2
    params(6) = 0.002;  // k3
    params(7) = -0.001;  // k4
    params(8) = 0.0007;  // p1
    params(9) = -0.0004;  // p2
    params(10) = 0.0003;  // sx1
    params(11) = -0.0002;  // sy1
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
      ModelT* model = dynamic_cast<ModelT*>(state.intrinsics[camera_index].get());
      ASSERT_TRUE(model);
      
      Matrix<double, Eigen::Dynamic, 1>& params = model->parameters();
      params(0) += 20 * ((rand() % 10000) / 5000.f - 1.f);
      params(1) += 20 * ((rand() % 10000) / 5000.f - 1.f);
      params(2) += 20 * ((rand() % 10000) / 5000.f - 1.f);
      params(3) += 20 * ((rand() % 10000) / 5000.f - 1.f);
      params(4) += 0.01 * ((rand() % 10000) / 5000.f - 1.f);
      params(5) += 0.005 * ((rand() % 10000) / 5000.f - 1.f);
      params(6) += 0.001 * ((rand() % 10000) / 5000.f - 1.f);
      params(7) += 0.001 * ((rand() % 10000) / 5000.f - 1.f);
      params(8) += 0.0005 * ((rand() % 10000) / 5000.f - 1.f);
      params(9) += 0.0005 * ((rand() % 10000) / 5000.f - 1.f);
      params(10) += 0.0005 * ((rand() % 10000) / 5000.f - 1.f);
      params(11) += 0.0005 * ((rand() % 10000) / 5000.f - 1.f);
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
  
  const double result_threshold = use_cuda ? 8.f : 1e-6f;  // TODO: Can we get the CUDA version to be more accurate?
  EXPECT_LE(last_cost, num_cameras * result_threshold);
}
}

TEST(CentralThinPrismFisheye, ProjectionAndUnprojection) {
  constexpr int kCameraWidth = 640 / 4;
  constexpr int kCameraHeight = 480 / 4;
  
  for (int i = 0; i < 2; ++ i) {
    bool use_equidistant_projection = (i == 0);
    CentralThinPrismFisheyeModel model(kCameraWidth, kCameraHeight, use_equidistant_projection);
    // NOTE: The field of view cannot be too large, otherwise the in-built
    //       fisheye distortion of CentralThinPrismFisheyeModel fails.
    model.parameters() <<
        kCameraHeight / 1.5,
        kCameraHeight / 1.5,
        kCameraWidth / 2,
        kCameraHeight / 2,
        0.02, -0.01, 0.002, -0.001,
        0.0002, -0.0001, 0.0003, -0.0004;
    
    TestProjectUnproject(model);
  }
}

TEST(CentralThinPrismFisheye, FitToDenseModel) {
  constexpr int kCameraWidth = 640 / 4;
  constexpr int kCameraHeight = 480 / 4;
  
  for (int i = 0; i < 2; ++ i) {
    bool use_equidistant_projection = (i == 0);
    
    // Define the ground truth model
    CentralThinPrismFisheyeModel model(kCameraWidth, kCameraHeight, use_equidistant_projection);
    model.parameters() <<
        kCameraHeight / 1.5,
        kCameraHeight / 1.5,
        kCameraWidth / 2,
        kCameraHeight / 2,
        0.04f, -0.03f, 0.02f, -0.01f,
        0.001f, -0.002f, 0.003f, -0.004f;
    Mat3d parametric_r_dense = AngleAxisd(0.01, Vec3d(1, 1, 1).normalized()).matrix();
    Mat3d dense_r_parametric = parametric_r_dense.transpose();
    
    Image<Vec3d> dense_model(model.width(), model.height());
    for (int y = 0; y < model.height(); ++ y) {
      for (int x = 0; x < model.width(); ++ x) {
        Vec3d direction;
        if (!model.Unproject(x + 0.5f, y + 0.5f, &direction)) {
          LOG(FATAL) << "Error in test setup (or CentralThinPrismFisheye::Unproject()): ground truth model values produce unprojection failure";
        }
        dense_model(x, y) = dense_r_parametric * direction;
        if (direction.hasNaN()) {
          LOG(FATAL) << "Error in test setup (or CentralThinPrismFisheye::Unproject()): ground truth model values produce NaN direction when unprojecting from image area";
        }
      }
    }
    
    CentralThinPrismFisheyeModel estimated_model(dense_model.width(), dense_model.height(), use_equidistant_projection);
    TestFitCentralModel(&estimated_model, dense_model, parametric_r_dense, model);
  }
}

TEST(CentralThinPrismFisheye, OptimizeJointly) {
  TestOptimizeJointlyParametric<CentralThinPrismFisheyeModel>(false, 1, /*use_equidistant_projection*/ true);
}

TEST(CentralThinPrismFisheye, OptimizeJointlyNoEquidistant) {
  TestOptimizeJointlyParametric<CentralThinPrismFisheyeModel>(false, 1, /*use_equidistant_projection*/ false);
}

TEST(CentralThinPrismFisheye, OptimizeJointlyRig) {
  TestOptimizeJointlyParametric<CentralThinPrismFisheyeModel>(false, 2, /*use_equidistant_projection*/ true);
}

TEST(CentralThinPrismFisheye, OptimizeJointlyRigNoEquidistant) {
  TestOptimizeJointlyParametric<CentralThinPrismFisheyeModel>(false, 2, /*use_equidistant_projection*/ false);
}

TEST(CentralThinPrismFisheye, LinearInitialization) {
  // Create a ground truth model
  CentralThinPrismFisheyeModel gt_model(640, 480, true);
  gt_model.parameters() <<
      320, 320, 320, 240, 0.02, -0.01, 0, 0, 0, 0, 0, 0;
  
  // Attempt to determine the non-zero ground truth model parameters from known
  // projection <-> unprojection pairs. Sample those pairs.
  constexpr int kNumPairs = 200;
  vector<Vec2f> projections(kNumPairs);
  vector<Vec2f> unprojections(kNumPairs);
  
  for (int i = 0; i < kNumPairs; ++ i) {
    projections[i] = Vec2f(gt_model.width(), gt_model.height()).cwiseProduct(0.5 * (Vec2f::Random() + Vec2f::Constant(1)));
    Vec3d direction;
    if (!gt_model.Unproject(projections[i].x(), projections[i].y(), &direction)) {
      // Try again.
      -- i;
      continue;
    }
    unprojections[i].x() = direction.x() / direction.z();
    unprojections[i].y() = direction.y() / direction.z();
    
    // Add some noise
    unprojections[i] += Vec2f::Random() * 0.001;
  }
  
  // Build the equation system.
  // Known: px, nx
  // Unknown: fx, cx, k1
  // 
  // px = fx * nx  +  fx * k1 * (nx * r2)  +  cx
  // 
  // --> Unknowns: fx, (fx * k1), cx
  // 
  // Equation system(s):
  // (nx nx*r2 nx*r4   0     0     0 1 0)   (   fx)   (px)
  // ( 0     0     0  ny ny*r2 ny*r4 0 1) * (fx_k1) = (py)
  //                                        (fx_k2)
  //                                        (   fy)
  //                                        (fy_k1)
  //                                        (fy_k2)
  //                                        (   cx)
  //                                        (   cy)
  Matrix<double, Eigen::Dynamic, 4> matrix_x;
  matrix_x.resize(kNumPairs, Eigen::NoChange);
  Matrix<double, Eigen::Dynamic, 4> matrix_y;
  matrix_y.resize(kNumPairs, Eigen::NoChange);
  
  Matrix<double, Eigen::Dynamic, 1> pixels_x;
  pixels_x.resize(kNumPairs, Eigen::NoChange);
  Matrix<double, Eigen::Dynamic, 1> pixels_y;
  pixels_y.resize(kNumPairs, Eigen::NoChange);
  
  u32 row_x = 0;
  u32 row_y = 0;
  for (int i = 0; i < kNumPairs; ++ i) {
    double r = unprojections[i].norm();
    
    Vec2d nxy;
    const double kEpsilon = static_cast<double>(1e-6);
    if (gt_model.use_equidistant_projection() && r > kEpsilon) {
      double theta_by_r = std::atan(r) / r;
      nxy.x() = theta_by_r * unprojections[i].coeff(0);
      nxy.y() = theta_by_r * unprojections[i].coeff(1);
    } else {
      nxy.x() = unprojections[i].coeff(0);
      nxy.y() = unprojections[i].coeff(1);
    }
    
    const double x2 = nxy.x() * nxy.x();
    const double y2 = nxy.y() * nxy.y();
    const double r2 = x2 + y2;
    const double r4 = r2 * r2;
    
    if (!std::isnan(nxy.x())) {
      matrix_x(row_x, 0) = nxy.x();
      matrix_x(row_x, 1) = nxy.x() * r2;
      matrix_x(row_x, 2) = nxy.x() * r4;
      matrix_x(row_x, 3) = 1;
      
      pixels_x(row_x, 0) = projections[i].x();
      
      ++ row_x;
    }
    if (!std::isnan(nxy.y())) {
      matrix_y(row_y, 0) = nxy.y();
      matrix_y(row_y, 1) = nxy.y() * r2;
      matrix_y(row_y, 2) = nxy.y() * r4;
      matrix_y(row_y, 3) = 1;
      
      pixels_y(row_y, 0) = projections[i].y();
      
      ++ row_y;
    }
  }
  
  matrix_x.conservativeResize(row_x, Eigen::NoChange);
  pixels_x.conservativeResize(row_x);
  
  matrix_y.conservativeResize(row_y, Eigen::NoChange);
  pixels_y.conservativeResize(row_y);
  
//   // Debug: Try to determine the rank of the matrices with SVD, checking the number of non-zero singular values
//   JacobiSVD<MatrixXd> svd_x(matrix_x, ComputeFullU | ComputeFullV);
//   LOG(INFO) << "svd_x singular values:\n" << svd_x.singularValues();
//   
//   JacobiSVD<MatrixXd> svd_y(matrix_y, ComputeFullU | ComputeFullV);
//   LOG(INFO) << "svd_y singular values:\n" << svd_y.singularValues();
  
  // Solve the (overdetermined) linear system of equations using the pseudoinverse
  Matrix<double, 4, 1> result_x = (matrix_x.transpose() * matrix_x).inverse() * (matrix_x.transpose() * pixels_x);
  Matrix<double, 4, 1> result_y = (matrix_y.transpose() * matrix_y).inverse() * (matrix_y.transpose() * pixels_y);
  
//   // Debug:
//   LOG(INFO) << "result_x:\n" << result_x;
//   LOG(INFO) << "gt_result_x:\n"
//             << gt_model.parameters()[0] << "\n"
//             << (gt_model.parameters()[0] * gt_model.parameters()[4]) << "\n"
//             << (gt_model.parameters()[0] * gt_model.parameters()[5]) << "\n"
//             << gt_model.parameters()[2];
//   LOG(INFO) << "";
//   LOG(INFO) << "result_y:\n" << result_y;
//   LOG(INFO) << "gt_result_y:\n"
//             << gt_model.parameters()[1] << "\n"
//             << (gt_model.parameters()[1] * gt_model.parameters()[4]) << "\n"
//             << (gt_model.parameters()[1] * gt_model.parameters()[5]) << "\n"
//             << gt_model.parameters()[3];
  
  // Extract the values of k1, k2 from the results.
  double k1 = 0.5 * ((result_x(1) / result_x(0)) + (result_y(1) / result_y(0)));
  double k2 = 0.5 * ((result_x(2) / result_x(0)) + (result_y(2) / result_y(0)));
  
  CHECK_LE(fabs(k1 - gt_model.parameters()[4]), 0.002);
  CHECK_LE(fabs(k2 - gt_model.parameters()[5]), 0.002);
}
