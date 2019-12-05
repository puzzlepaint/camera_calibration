// Copyright 2017, 2019 ETH Zürich, Thomas Schöps
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


#include "libvis/logging.h"
#include <gtest/gtest.h>

#include "libvis/camera.h"

using namespace vis;

namespace {

template <class CameraT>
void TestUnprojectProjectIsIdentity(const CameraT& test_camera) {
  constexpr float kEpsilon = 1e-5;
  
  // Pixel center convention.
  {
    Vec2f pixel_coordinate(1, 2);
    
    Vec3f pixel_direction = test_camera.UnprojectFromPixelCenterConv(pixel_coordinate.cast<typename CameraT::ScalarT>()).template cast<float>();
    
    Vec2f pixel_coordinate_reprojected = test_camera.ProjectToPixelCenterConv(pixel_direction.cast<typename CameraT::ScalarT>()).template cast<float>();
    EXPECT_NEAR(pixel_coordinate.x(), pixel_coordinate_reprojected.x(), kEpsilon);
    EXPECT_NEAR(pixel_coordinate.y(), pixel_coordinate_reprojected.y(), kEpsilon);
    
    Matrix<typename CameraT::ScalarT, 2, 1> pixel_coordinate_reprojected_scalar;
    bool is_visible = test_camera.ProjectToPixelCenterConvIfVisible(
        pixel_direction.cast<typename CameraT::ScalarT>(), /*pixel_border*/ 0, &pixel_coordinate_reprojected_scalar);
    EXPECT_TRUE(is_visible);
    if (is_visible) {
      EXPECT_NEAR(pixel_coordinate.x(), pixel_coordinate_reprojected_scalar.x(), kEpsilon);
      EXPECT_NEAR(pixel_coordinate.y(), pixel_coordinate_reprojected_scalar.y(), kEpsilon);
    }
  }
  
  // Pixel corner convention.
  {
    Vec2f pixel_coordinate(1.5, 2.5);
    
    Vec3f pixel_direction = test_camera.UnprojectFromPixelCornerConv(pixel_coordinate.cast<typename CameraT::ScalarT>()).template cast<float>();
    
    Vec2f pixel_coordinate_reprojected = test_camera.ProjectToPixelCornerConv(pixel_direction.cast<typename CameraT::ScalarT>()).template cast<float>();
    EXPECT_NEAR(pixel_coordinate.x(), pixel_coordinate_reprojected.x(), kEpsilon);
    EXPECT_NEAR(pixel_coordinate.y(), pixel_coordinate_reprojected.y(), kEpsilon);
    
    Matrix<typename CameraT::ScalarT, 2, 1> pixel_coordinate_reprojected_scalar;
    bool is_visible = test_camera.ProjectToPixelCornerConvIfVisible(
        pixel_direction.cast<typename CameraT::ScalarT>(), /*pixel_border*/ 0, &pixel_coordinate_reprojected_scalar);
    EXPECT_TRUE(is_visible);
    if (is_visible) {
      EXPECT_NEAR(pixel_coordinate.x(), pixel_coordinate_reprojected_scalar.x(), kEpsilon);
      EXPECT_NEAR(pixel_coordinate.y(), pixel_coordinate_reprojected_scalar.y(), kEpsilon);
    }
  }
  
  // Ratio convention.
  {
    Vec2f pixel_coordinate(1.5 / test_camera.width(), 2.5 / test_camera.height());
    
    Vec3f pixel_direction = test_camera.UnprojectFromRatioConv(pixel_coordinate.cast<typename CameraT::ScalarT>()).template cast<float>();
    
    Vec2f pixel_coordinate_reprojected = test_camera.ProjectToRatioConv(pixel_direction.cast<typename CameraT::ScalarT>()).template cast<float>();
    EXPECT_NEAR(pixel_coordinate.x(), pixel_coordinate_reprojected.x(), kEpsilon);
    EXPECT_NEAR(pixel_coordinate.y(), pixel_coordinate_reprojected.y(), kEpsilon);
    
    Matrix<typename CameraT::ScalarT, 2, 1> pixel_coordinate_reprojected_scalar;
    bool is_visible = test_camera.ProjectToRatioConvIfVisible(
        pixel_direction.cast<typename CameraT::ScalarT>(), /*pixel_border*/ 0, &pixel_coordinate_reprojected_scalar);
    EXPECT_TRUE(is_visible);
    if (is_visible) {
      EXPECT_NEAR(pixel_coordinate.x(), pixel_coordinate_reprojected_scalar.x(), kEpsilon);
      EXPECT_NEAR(pixel_coordinate.y(), pixel_coordinate_reprojected_scalar.y(), kEpsilon);
    }
  }
}

}

// Tests that unprojection followed by projection equals the identity function.
TEST(Camera, UnprojectProjectIsIdentity) {
  LOG(INFO) << "Model: PinholeCamera4f";
  float pinhole_parameters[4] = {120, 120, 120, 120};  // fx, fy, cx, cy.
  PinholeCamera4f pinhole_camera(240, 240, pinhole_parameters);
  TestUnprojectProjectIsIdentity(pinhole_camera);
  
  LOG(INFO) << "Model: RadtanCamera8d";
  double radtan8_parameters[8] = {0.01, 0.02, 0.003, 0.002, 120, 120, 120, 120};
  RadtanCamera8d radtan8_camera(240, 240, radtan8_parameters);
  TestUnprojectProjectIsIdentity(radtan8_camera);
  
  LOG(INFO) << "Model: RadtanCamera9d";
  double radtan9_parameters[9] = {0.01, 0.02, -0.015, 0.003, 0.002, 120, 120, 120, 120};
  RadtanCamera9d radtan9_camera(240, 240, radtan9_parameters);
  TestUnprojectProjectIsIdentity(radtan9_camera);
  
  LOG(INFO) << "Model: ThinPrismFisheyeCamera12d";
  double thin_prism_fisheye_parameters[12] = {0.01, 0.02, -0.024, 0.003, 0.002, -0.001, 0.005, -0.006, 120, 120, 120, 120};
  ThinPrismFisheyeCamera12d thin_prism_fisheye_camera(240, 240, thin_prism_fisheye_parameters);
  TestUnprojectProjectIsIdentity(thin_prism_fisheye_camera);
}

// Tests that identity scaling does not change camera parameters.
TEST(Camera, IdentityScaling) {
  float parameters[4] = {120, 121, 122, 123};  // fx, fy, cx, cy.
  PinholeCamera4f test_camera(240, 242, parameters);
  
  PinholeCamera4f* scaled_camera = test_camera.Scaled(1);
  EXPECT_EQ(test_camera.width(), scaled_camera->width());
  EXPECT_EQ(test_camera.height(), scaled_camera->height());
  EXPECT_FLOAT_EQ(test_camera.parameters()[0], scaled_camera->parameters()[0]);
  EXPECT_FLOAT_EQ(test_camera.parameters()[1], scaled_camera->parameters()[1]);
  EXPECT_FLOAT_EQ(test_camera.parameters()[2], scaled_camera->parameters()[2]);
  EXPECT_FLOAT_EQ(test_camera.parameters()[3], scaled_camera->parameters()[3]);
  delete scaled_camera;
}
