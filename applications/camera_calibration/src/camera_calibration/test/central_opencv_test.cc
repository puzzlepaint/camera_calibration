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
#include "camera_calibration/models/central_opencv.h"
#include "camera_calibration/test/util.h"

using namespace vis;

TEST(CentralOpenCV, ProjectionAndUnprojection) {
  constexpr int kCameraWidth = 640 / 4;
  constexpr int kCameraHeight = 480 / 4;
  
  CentralOpenCVModel model(kCameraWidth, kCameraHeight);
  model.parameters() <<
      kCameraHeight / 1.5,
      kCameraHeight / 1.5,
      kCameraWidth / 2,
      kCameraHeight / 2,
      0.001, 0.002, -0.0001, 0.0001,
      -0.0002, 0.0001, 0.00012, -0.00012;
  
  TestProjectUnproject(model);
}

TEST(CentralOpenCV, FitToDenseModel) {
  constexpr int kCameraWidth = 640 / 4;
  constexpr int kCameraHeight = 480 / 4;
  
  // Define the ground truth model
  CentralOpenCVModel model(kCameraWidth, kCameraHeight);
  model.parameters() <<
      kCameraHeight * 1.1,
      kCameraHeight * 1.1,
      kCameraWidth / 2,
      kCameraHeight / 2,
      0.04f, -0.03f, 0.02f, -0.01f,
      0.001f, -0.002f, 0.003f, -0.004f;
  Mat3d parametric_r_dense = AngleAxisd(0.01, Vec3d(1, 1, 1).normalized()).matrix();
  Mat3d dense_r_parametric = parametric_r_dense.transpose();
  
  // Image<Vec3u8> model_visualization;
  // VisualizeModelDirections(model, &model_visualization);
  // ImageDisplay model_visualization_display;
  // model_visualization_display.Update(model_visualization, "Ground truth model");
  
  Image<Vec3d> dense_model(model.width(), model.height());
  for (int y = 0; y < model.height(); ++ y) {
    for (int x = 0; x < model.width(); ++ x) {
      Vec3d direction;
      if (!model.Unproject(x + 0.5f, y + 0.5f, &direction)) {
        LOG(FATAL) << "Error in test setup (or CentralOpenCV::Unproject()): ground truth model values produce unprojection failure";
      }
      dense_model(x, y) = dense_r_parametric * direction;
      if (direction.hasNaN()) {
        LOG(FATAL) << "Error in test setup (or CentralOpenCV::Unproject()): ground truth model values produce NaN direction when unprojecting from image area";
      }
    }
  }
  
  CentralOpenCVModel estimated_model(dense_model.width(), dense_model.height());
  TestFitCentralModel(&estimated_model, dense_model, parametric_r_dense, model);
}
