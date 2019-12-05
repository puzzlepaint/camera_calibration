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

#include "camera_calibration/calibration.h"
#include "camera_calibration/models/central_generic.h"
#include "camera_calibration/util.h"

using namespace vis;

// Tests that ChooseNiceCameraOrientation() chooses a unique orientation,
// regardless of the initial orientation. I.e., calibrating the same camera
// multiple times with multiple datasets should yield exactly the same result,
// not a somehow rotated representation.
TEST(CameraUniqueRepresentation, CentralGeneric) {
  srand(0);
  
  constexpr int kCameraWidth = 640;
  constexpr int kCameraHeight = 480;
  
  CentralGenericModel model(
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
  
  // Call ChooseNiceCameraOrientation() to get to the canonical state
  model.ChooseNiceCameraOrientation();
  
  // Disturb the state and make sure that ChooseNiceCameraOrientation() returns
  // to the same canonical state as before
  for (int test = 0; test < 10; ++ test) {
    // Disturb
    Mat3d rotation = SO3d::exp(SO3d::Tangent::Random()).matrix();
    CentralGenericModel rotated_model(model);
    rotated_model.Rotate(rotation);
    
    // Correct
    rotated_model.ChooseNiceCameraOrientation();
    
    // Compare
    for (u32 y = 0; y < grid.height(); ++ y) {
      for (u32 x = 0; x < grid.width(); ++ x) {
        EXPECT_LE((model.grid()(x, y) - rotated_model.grid()(x, y)).norm(), 1e-5f);
      }
    }
  }
}
