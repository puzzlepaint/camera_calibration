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
#include "camera_calibration/models/central_generic.h"
#include "camera_calibration/test/util.h"
#include "camera_calibration/util.h"

using namespace vis;

TEST(CentralGenericBSpline, ProjectUnproject) {
  TestProjectUnproject<CentralGenericModel>();
}

TEST(CentralGenericBSpline, ModelOptimization) {
  TestModelOptimization<CentralGenericModel>();
}

TEST(CentralGenericBSpline, OptimizeJointly) {
  TestOptimizeJointly<CentralGenericModel>(false, SchurMode::Dense, 1);
}

TEST(CentralGenericBSpline, OptimizeJointlyRig) {
  TestOptimizeJointly<CentralGenericModel>(false, SchurMode::Dense, 2);
}

TEST(CentralGenericBSpline, CUDAOptimizeJointly) {
  TestOptimizeJointly<CentralGenericModel>(true, SchurMode::Dense, 1);
}

TEST(CentralGenericBSpline, CUDAOptimizeJointlyRig) {
  TestOptimizeJointly<CentralGenericModel>(true, SchurMode::Dense, 2);
}


TEST(SchurMode, DenseCUDA) {
  TestOptimizeJointly<CentralGenericModel>(false, SchurMode::DenseCUDA, 1);
}

TEST(SchurMode, DenseOnTheFly) {
  TestOptimizeJointly<CentralGenericModel>(false, SchurMode::DenseOnTheFly, 1);
}

TEST(SchurMode, Sparse) {
  TestOptimizeJointly<CentralGenericModel>(false, SchurMode::Sparse, 1);
}

TEST(SchurMode, SparseOnTheFly) {
  TestOptimizeJointly<CentralGenericModel>(false, SchurMode::SparseOnTheFly, 1);
}
