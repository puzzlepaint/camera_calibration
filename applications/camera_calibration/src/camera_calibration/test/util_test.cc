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

#include "camera_calibration/util.h"

using namespace vis;

TEST(Util, DeterminePointCloudRotation) {
  vector<Vec3d> src_points;
  src_points.push_back(Vec3d(1, 2, 3));
  src_points.push_back(Vec3d(3, 2, 1));
  src_points.push_back(Vec3d(1, 1, 2));
  src_points.push_back(Vec3d(4, 2, 2));
  src_points.push_back(Vec3d(2, 2, 1));
  
  SO3d ground_truth_dest_R_src =
      SO3d(Quaterniond(AngleAxisd(0.42f, Vec3d(1, 3, 2).normalized())));
  
  vector<Vec3d> dest_points;
  dest_points.resize(src_points.size());
  for (usize i = 0; i < src_points.size(); ++ i) {
    dest_points[i] = ground_truth_dest_R_src * src_points[i];
  }
  
  Mat3d dest_r_src = DeterminePointCloudRotation(dest_points, src_points);
  
  EXPECT_LT((dest_r_src - ground_truth_dest_R_src.matrix()).norm(), 1e-5f);
}
