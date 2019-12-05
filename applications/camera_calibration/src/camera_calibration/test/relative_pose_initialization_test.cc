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


#include <libvis/eigen.h>
#include <libvis/libvis.h>
#include <libvis/logging.h>
#include <libvis/sophus.h>
#include <gtest/gtest.h>

#include "camera_calibration/relative_pose_initialization/algorithms.h"
#include "camera_calibration/util.h"

using namespace vis;

static bool TestRelativePoseInitialization(
    int num_points,
    int num_test_cases,
    bool planar_pattern,
    bool central_camera,
    string algorithm) {
  // TODO: Adapt to chosen algorithm?
  constexpr int kNumImages = 3;
  
  // All calibration target points are sampled on the z-plane if assuming
  // a planar pattern. This choice of plane is required by the algorithms which
  // assume a planar pattern!
  Eigen::Hyperplane<float, 3> z_plane =
      Eigen::Hyperplane<float, 3>::Through(
          Vec3f(0, 0, 0),
          Vec3f(1, 0, 0),
          Vec3f(0, 1, 0));
  
  // Allocate camera poses and point clouds
  SE3f global_tr_camera[kNumImages];
  Point3fCloud clouds[kNumImages];  // indexed by [image_index][point_index].
  for (int image = 0; image < kNumImages; ++ image) {
    clouds[image].Resize(num_points);
  }
  
  int num_fails = 0;
  for (int test_case = 0; test_case < num_test_cases; ++ test_case) {
    // Generate random camera poses as slight distortions of the identity
    // pose, being close to (0, 0, -5).
    for (int image = 0; image < kNumImages; ++ image) {
      global_tr_camera[image] = SE3f::exp(0.1 * SE3f::Tangent::Random());
      global_tr_camera[image].translation().z() -= 5;
    }
    
    // Generate synthetic matching points on the calibration target.
    for (int pixel = 0; pixel < num_points; ++ pixel) {
      // Generate ray direction
      Vec3f direction;
      while (true) {
        direction = Vec3f::Random();
        if (direction.squaredNorm() < 0.001) {
          continue;
        }
        
        // // For camera looking in the +z half space only
        // if (direction.z() < 0) {
        //   direction.z() = -direction.z();
        // }
        
        // For a camera looking clearly towards +z
        if (direction.z() < 0.2) {
          continue;
        }
        
        direction.normalize();
        break;
      }
      
      // Sample the calibration target points on the ray from each camera pose.
      Vec3f offset;
      if (central_camera) {
        offset = Vec3f::Zero();
      } else {
        offset = Vec3f::Random();
      }
      
      Eigen::Hyperplane<float, 3> plane;
      if (planar_pattern) {
        plane = z_plane;
      } else {
        Vec3f normal;
        while (true) {
          normal = Vec3f::Random();
          if (normal.squaredNorm() < 0.001) {
            continue;
          }
          normal.normalize();
          break;
        }
        plane = Eigen::Hyperplane<float, 3>(normal, 4 * ((rand() % 10000) / 10001.f - 0.5));
      }
      
      for (int image = 0; image < kNumImages; ++ image) {
        Eigen::ParametrizedLine<float, 3> line =
            Eigen::ParametrizedLine<float, 3>::Through(global_tr_camera[image] * (offset),
                                                       global_tr_camera[image] * (direction + offset));
        clouds[image][pixel].position() = line.intersectionPoint(plane);
      }
    }
    
    int used_clouds = -1;
    SE3d cloudX_tr_cloud[2];
    bool optical_center_estimated = false;
    Vec3d optical_center = Vec3d::Constant(numeric_limits<double>::quiet_NaN());
    
    if (algorithm == "ramalingam_noncentral_camera_3d_target") {
      used_clouds = 3;
      if (!NonCentralCamera3DCalibrationObjectRelativePose(clouds, cloudX_tr_cloud)) {
        std::cout << "FAIL: NonCentralCamera3DCalibrationObjectRelativePose() failed." << endl;
        ++ num_fails;
        continue;
      }
    } else if (algorithm == "ramalingam_noncentral_camera_planar_target") {
      // DEBUG:
      SE3d GT_cloud2_tr_cloud[2];
      GT_cloud2_tr_cloud[0] = (global_tr_camera[2] * global_tr_camera[0].inverse()).cast<double>();
      GT_cloud2_tr_cloud[1] = (global_tr_camera[2] * global_tr_camera[1].inverse()).cast<double>();
      
      used_clouds = 3;
      if (!NonCentralCameraPlanarCalibrationObjectRelativePose(clouds, cloudX_tr_cloud, GT_cloud2_tr_cloud)) {
        std::cout << "FAIL: NonCentralCameraPlanarCalibrationObjectRelativePose() failed." << endl;
        ++ num_fails;
        continue;
      }
    } else if (algorithm == "ramalingam_central_camera_3d_target") {
      used_clouds = 2;
      if (!CentralCamera3DCalibrationObjectRelativePose(clouds, cloudX_tr_cloud, &optical_center)) {
        std::cout << "FAIL: CentralCamera3DCalibrationObjectRelativePose() failed." << endl;
        ++ num_fails;
        continue;
      }
      optical_center_estimated = true;
    } else if (algorithm == "ramalingam_central_camera_planar_target") {
      used_clouds = 3;
      if (!CentralCameraPlanarCalibrationObjectRelativePose(clouds, cloudX_tr_cloud, &optical_center)) {
        std::cout << "FAIL: CentralCameraPlanarCalibrationObjectRelativePose() failed." << endl;
        ++ num_fails;
        continue;
      }
      optical_center_estimated = true;
    } else {
      LOG(ERROR) << "Unknown algorithm: " << algorithm;
      return false;
    }
    
    // Verify the results.
    constexpr float kEpsilon = 1e-2f;  // TODO: Make the algorithms more numerically stable, then reduce this
    
    if (used_clouds == 3) {
      SE3d GT_cloud2_tr_cloud0 = (global_tr_camera[2] * global_tr_camera[0].inverse()).cast<double>();
      SE3d GT_cloud2_tr_cloud1 = (global_tr_camera[2] * global_tr_camera[1].inverse()).cast<double>();
      
      bool have_failure = false;
      
      if (!((GT_cloud2_tr_cloud0.log() - cloudX_tr_cloud[0].log()).norm() <= kEpsilon)) {  // treat NaNs as failure
        std::cout << "FAIL: cloud2_tr_cloud0 is wrong" << endl;
        std::cout << "Diff rotation:" << endl << (cloudX_tr_cloud[0].rotationMatrix() - GT_cloud2_tr_cloud0.rotationMatrix()) << endl;
        std::cout << "Diff translation:" << endl << (cloudX_tr_cloud[0].translation() - GT_cloud2_tr_cloud0.translation()) << endl;
        if (!have_failure) {
          have_failure = true;
          ++ num_fails;
        }
      }
      
      if (!((GT_cloud2_tr_cloud1.log() - cloudX_tr_cloud[1].log()).norm() <= kEpsilon)) {  // treat NaNs as failure
        std::cout << "FAIL: cloud2_tr_cloud1 is wrong" << endl;
        std::cout << "Diff rotation:" << endl << (cloudX_tr_cloud[1].rotationMatrix() - GT_cloud2_tr_cloud1.rotationMatrix()) << endl;
        std::cout << "Diff translation:" << endl << (cloudX_tr_cloud[1].translation() - GT_cloud2_tr_cloud1.translation()) << endl;
        if (!have_failure) {
          have_failure = true;
          ++ num_fails;
        }
      }
      
      if (optical_center_estimated) {
        EXPECT_LE((optical_center - global_tr_camera[2].translation().cast<double>()).norm(), 2e-3f);
      }
    } else if (used_clouds == 2) {
      SE3d GT_cloud1_tr_cloud0 = (global_tr_camera[1] * global_tr_camera[0].inverse()).cast<double>();
      
      if (!((GT_cloud1_tr_cloud0.log() - cloudX_tr_cloud[0].log()).norm() <= kEpsilon)) {  // treat NaNs as failure
        std::cout << "FAIL: cloud1_tr_cloud0 is wrong" << endl;
        std::cout << "Diff rotation:" << endl << (cloudX_tr_cloud[0].rotationMatrix() - GT_cloud1_tr_cloud0.rotationMatrix()) << endl;
        std::cout << "Diff translation:" << endl << (cloudX_tr_cloud[0].translation() - GT_cloud1_tr_cloud0.translation()) << endl;
        ++ num_fails;
      }
      
      if (optical_center_estimated) {
        EXPECT_LE((optical_center - global_tr_camera[1].translation().cast<double>()).norm(), 2e-3f);
      }
    } else {
      LOG(ERROR) << "This case should not occur.";
      return false;
    }
    
    // if (num_fails == num_fails_pre) {
    //   std::cout << "Test instance successful!\n";
    // }
  }
  
  EXPECT_EQ(0, num_fails);
  return true;
}

TEST(RelativePoseInitialization, RamalingamCentralCameraPlanarTarget) {
  srand(0);
  TestRelativePoseInitialization(
      /*num_points*/ 30,  // min: 4 (but this will lead to some failures due to inaccuracies)
      /*num_test_cases*/ 30,
      /*planar_pattern*/ true,
      /*central_camera*/ true,
      "ramalingam_central_camera_planar_target");
}

TEST(RelativePoseInitialization, RamalingamCentralCamera3DTarget) {
  srand(0);
  TestRelativePoseInitialization(
      /*num_points*/ 30,  // min: 10
      /*num_test_cases*/ 30,
      /*planar_pattern*/ false,
      /*central_camera*/ true,
      "ramalingam_central_camera_3d_target");
}

TEST(RelativePoseInitialization, RamalingamNoncentralCameraPlanarTarget) {
  srand(0);
  TestRelativePoseInitialization(
      /*num_points*/ 30,  // min: 11
      /*num_test_cases*/ 30,
      /*planar_pattern*/ true,
      /*central_camera*/ false,
      "ramalingam_noncentral_camera_planar_target");
}

TEST(RelativePoseInitialization, RamalingamNoncentralCamera3DTarget) {
  srand(0);
  TestRelativePoseInitialization(
      /*num_points*/ 30,  // min: 26
      /*num_test_cases*/ 30,
      /*planar_pattern*/ false,
      /*central_camera*/ false,
      "ramalingam_noncentral_camera_3d_target");
}
