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
#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac/SampleConsensusProblem.hpp>
#include <opengv/sac/SampleConsensus.hpp>
#include <opengv/sac_problems/point_cloud/PointCloudSacProblem.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>

#include "camera_calibration/util.h"

using namespace vis;

/// Tests whether we can successfully use OpenGV (or whether it crashes). It
/// may crash due to incompatible compiler flags (-march=native).
TEST(OpenGV, AbsolutePoseSacProblem) {
  // Generate random points
  constexpr int kNumPoints = 30;
  opengv::points_t global_points(kNumPoints);
  for (int i = 0; i < kNumPoints; ++ i) {
    global_points[i] = Vec3d::Random();
  }
  
  // Generate random ground truth camera pose
  SE3d camera_tr_global = SE3d::exp(0.1 * SE3d::Tangent::Random());
  camera_tr_global.translation().z() += 5;
  
  // Compute bearing vectors
  opengv::bearingVectors_t bearing_vectors(kNumPoints);
  for (int i = 0; i < kNumPoints; ++ i) {
    Vec3d camera_point = camera_tr_global * global_points[i];  // NOTE: (SE3 * point) is inefficient! Should convert the quaternion to a matrix first.
    bearing_vectors[i] = camera_point.normalized();
  }
  
  // Estimate the camera pose using OpenGV
  CHECK_EQ(bearing_vectors.size(), global_points.size());
  opengv::absolute_pose::CentralAbsoluteAdapter adapter(
      bearing_vectors,
      global_points);
  
  std::shared_ptr<opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem> absposeproblem_ptr(
      new opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem(
          adapter, opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem::KNEIP));
  
  opengv::sac::Ransac<opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem> ransac;
  ransac.sac_model_ = absposeproblem_ptr;
  ransac.threshold_ = 1.0 - cos(atan(3 / 720.0));  // Equals a reprojection error of 3 pixels for focal length 720.
  ransac.max_iterations_ = 10;
  
  ASSERT_TRUE(ransac.computeModel());
  
  // Non-linear optimization (using all correspondences)
  adapter.sett(ransac.model_coefficients_.block<3, 1>(0, 3));
  adapter.setR(ransac.model_coefficients_.block<3, 3>(0, 0));
  opengv::transformation_t global_tr_camera_matrix = opengv::absolute_pose::optimize_nonlinear(adapter);
  SE3d global_tr_camera_estimate = SE3d(
      Sophus::SE3d(global_tr_camera_matrix.block<3, 3>(0, 0).cast<double>(),
                   global_tr_camera_matrix.block<3, 1>(0, 3).cast<double>()));
  
  EXPECT_LE(SE3d::log(camera_tr_global * global_tr_camera_estimate).norm(), 1e-4f);
}
