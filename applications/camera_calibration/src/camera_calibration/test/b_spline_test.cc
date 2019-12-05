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
#include <libvis/image_display.h>
#include <libvis/libvis.h>
#include <libvis/logging.h>
#include <libvis/sophus.h>
#include <gtest/gtest.h>

#include "camera_calibration/b_spline.h"

using namespace vis;

TEST(BSpline, SlowFastAlgorithmConsistency) {
  Vec2f control_point_data[] = {
      Vec2f(0, 0), Vec2f(0, 0), Vec2f(0, 0), Vec2f(0, 0),
      Vec2f(0, 0), Vec2f(1, 1), Vec2f(2, 2), Vec2f(3, 3),
      Vec2f(0, 0), Vec2f(4, 4), Vec2f(5, 5), Vec2f(6, 6),
      Vec2f(0, 0), Vec2f(7, 7), Vec2f(8, 8), Vec2f(9, 9)};
  Image<Vec2f> control_points(4, 4, control_point_data);
  
  Image<Vec3u8> visualization(500, 100);
  visualization.SetTo(Vec3u8(255, 255, 255));
  for (int i = 0; i < 500; ++ i) {
    Vec2f result = EvalUniformCubicBSplineSurfaceGenericSlow(control_points, 1. + i / 500., 1.5);
    Vec2f result2 = EvalUniformCubicBSplineSurface(control_points, 1. + i / 500., 1.5);
    
    EXPECT_NEAR(result.x(), result2.x(), 1e-5f);
    EXPECT_NEAR(result.y(), result2.y(), 1e-5f);
  }
}
